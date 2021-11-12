--[[
  Wish to use a single state machine instantiation for client and server
  Intend to achieve this by inverting the read/writes to the shared bitmap
  on one of inbox/outbox on one of client/server
  This should be structured so duplex operation is possible, with client/server
  a specific instantiation of this. That is, instead of invoke/handle, the
  method should be on_lo (for exec when in == out == 0) and on_hi (for exec when
  in == out == 1)
  In/Outbox cross wired as before
  A correct state transition graph is one where at most one of client/server calls
  at most one on_lo/on_hi in a given time slice (exclusion), where it'll repeat itself
  after N operations. A useful one also executes in the desired sequence, alternating
  client/server and on_lo/on_hi. The correct one also starts at the right point in the
  sequence, which should be determined by starting conditions and which mailbox operations
  are inverted relative to memory
]]

--[[
-- Example runs.  Leading 0/0 means shared memory is zero initialised to begin.
-- Preferred because neither inverts data before writing

-- Invert reads from client inbox to get server first to execute
[0 0 true false false false] => success [15]
  [S] passed inbox 0 outbox 0  -> S_on_lo
  [C] passed inbox 0 outbox 0  -> C_on_lo
  [S] passed inbox 1 outbox 1  -> S_on_hi
  [C] passed inbox 1 outbox 1  -> C_on_hi

-- Invert reads from server inbox to get client first to execute
[0 0 false false true false] => success [15]
  [C] passed inbox 0 outbox 0  -> C_on_lo
  [S] passed inbox 0 outbox 0  -> S_on_lo
  [C] passed inbox 1 outbox 1  -> C_on_hi
  [S] passed inbox 1 outbox 1  -> S_on_hi

-- In both sequences the first execution is on mailbox [0 0] and the second on [1 1]
-- Client/server alternate, exclusion and progress provided
-- E.g. invert client inbox and set
--  server on_lo -> collect garbage
--  server on_hi -> do work
--  client on_lo -> submit work
--  client on_hi -> retrieve results
-- will run the server process first to initialise memory then match the current [0 0]
-- client submit work pattern.

-- different setups have results like client running on_lo then on_hi and server the
-- opposite way around which is more confusing than consistent order on the naming
-- Could refer to them as 'first' and 'second', or 'submit' and 'receive' consistently
-- if choosing a setup which has that consistency.
]]

local inspect = require 'inspect'

local function invert(s)
   assert((s == 0) or (s == 1))
   if s == 0 then
      return 1
   else
      return 0
   end
end

local function transition(state, describe)
   local inbox = state.inbox
   local outbox = state.outbox
   local call = 'none'

   assert ((inbox == 0) or (inbox == 1))
   assert ((outbox == 0) or (outbox == 1))

   local header = string.format('  [%s] passed inbox %s outbox %s', state.name, inbox, outbox)
   local changed = false
   
   if inbox == 0 then
      if outbox == 0 then
         -- 00
         call = 'on_lo'
         outbox = 1
         changed = true
      else
         -- 01

      end
   else
      if outbox == 0 then
         -- 10
      else
         -- 11
         call = 'on_hi'
         outbox = 0
         changed = true
      end
   end

   local footer = string.format('  -> %s_%s\n',state.name, call)
   
   if describe and changed then
      io.write (header..footer)
   end

   return {inbox = inbox,
           outbox = outbox,
           call = call,
   }
end

local state = {
   mem_lhs = 0,
   mem_rhs = 0,
}

local function state_equal(l, r)
   return (l.mem_lhs == r.mem_lhs) and (l.mem_rhs == r.mem_rhs) 
end

local function make_transition(name, inbox_state, inbox_invert, outbox_state, outbox_invert, describe)
   assert((inbox_invert == true) or (inbox_invert == false))
   assert((outbox_invert == true) or (outbox_invert == false))
   assert((inbox_state == 'mem_lhs') or (inbox_state == 'mem_rhs'))
   assert((outbox_state == 'mem_lhs') or (outbox_state == 'mem_rhs'))
   assert(inbox_state ~= outbox_state)

   local inbox_invert_func = (function (s) return s end)
   if inbox_invert then
      inbox_invert_func = invert
   end
   local outbox_invert_func = (function (s) return s end)
   if outbox_invert then
      outbox_invert_func = invert
   end
   
   local function f(state)
      local t = transition({name = name,
                            inbox = inbox_invert_func(state[inbox_state]),
                            outbox = outbox_invert_func(state[outbox_state]),
      }, describe)

      local r = {}     
      r[inbox_state] = inbox_invert_func(t.inbox)
      r[outbox_state] = outbox_invert_func(t.outbox)
      return r
   end
   
   return f
end


local function call(state, call_lhs, call_rhs)
   local lhs = call_lhs(state)
   local rhs = call_rhs(state)

   -- at least one function must be a no-op for every state
   local lhs_changed = not state_equal(state, lhs)
   local rhs_changed = not state_equal(state, rhs)

   local progress = rhs_changed or lhs_changed
   local exclusion = rhs_changed ~= lhs_changed
   local lhs_fixpoint = state_equal(lhs, call_lhs(lhs))
   local rhs_fixpoint = state_equal(rhs, call_rhs(rhs))

   -- requirements on implementation of transition
   assert(lhs_fixpoint and rhs_fixpoint)
   
   if not progress then
      return 'no-progress'
   end
   if not exclusion then
      return 'no-exclusion'
   end
   
   if (state_equal(lhs,state)) then
      return rhs
   else
      return lhs
   end
end

local function evaluate_transition(lhs_init,
                                   rhs_init,
                                   lhs_inbox_invert,
                                   lhs_outbox_invert,
                                   rhs_inbox_invert,
                                   rhs_outbox_invert,
                                   describe)
   assert((lhs_init == 0) or (lhs_init == 1))
   assert((rhs_init == 0) or (rhs_init == 1))
   local initial_state = {
      mem_lhs = lhs_init,
      mem_rhs = rhs_init,
   }

   local call_lhs = make_transition('C', 'mem_lhs', lhs_inbox_invert, 'mem_rhs', lhs_outbox_invert, describe)
   local call_rhs = make_transition('S', 'mem_rhs', rhs_inbox_invert, 'mem_lhs', rhs_outbox_invert, describe)

   local state = {
      mem_lhs = initial_state.mem_lhs,
      mem_rhs = initial_state.mem_rhs,
   }
   for i = 1, 8 do
      local r = call(state, call_lhs, call_rhs)
      if (type(r) ~= "table") then
         return r
      end
      if state_equal (r, initial_state) then
         return "success"
      end
      state = r
   end
   
   return 'too many transitions'
end

local function rank(lhs_init,
                    rhs_init,
                    lhs_inbox_invert,
                    lhs_outbox_invert,
                    rhs_inbox_invert,
                    rhs_outbox_invert)

   local r = 0

   -- like zero init
   if lhs_init == 0 and
      rhs_init == 0 then
      r = r + 10
   end

   if lhs_inbox_invert == false then
      r = r + 1
   end
   if rhs_inbox_invert == false then
      r = r + 1
   end

   if lhs_outbox_invert == false then
      r = r + 2
   end
   if rhs_outbox_invert == false then
      r = r + 2
   end

   return r
end


local function summarise(lhs_init,
                         rhs_init,
                         lhs_inbox_invert,
                         lhs_outbox_invert,
                         rhs_inbox_invert,
                         rhs_outbox_invert,
                         result)
   local r = rank(lhs_init,
                  rhs_init,
                  lhs_inbox_invert,
                  lhs_outbox_invert,
                  rhs_inbox_invert,
                  rhs_outbox_invert)

   print(string.format("[%s %s %s %s %s %s] => %s [%s]",
                       lhs_init, rhs_init, lhs_inbox_invert, lhs_outbox_invert, rhs_inbox_invert, rhs_outbox_invert,
                       result, r))
end


for lhs_init = 0, 1 do
for rhs_init = 0, 1 do
for _,lhs_inbox_invert in pairs({true, false}) do
for _,lhs_outbox_invert in pairs({true, false}) do
for _,rhs_inbox_invert in pairs({true, false}) do
for _,rhs_outbox_invert in pairs({true, false}) do

   local e = evaluate_transition(lhs_init, rhs_init, lhs_inbox_invert, lhs_outbox_invert, rhs_inbox_invert, rhs_outbox_invert, false)
   local r = rank(lhs_init, rhs_init, lhs_inbox_invert, lhs_outbox_invert, rhs_inbox_invert, rhs_outbox_invert)

   local r_req = 14
   if e == "success" and r >= r_req then
      summarise(lhs_init, rhs_init, lhs_inbox_invert, lhs_outbox_invert, rhs_inbox_invert, rhs_outbox_invert, e)
      evaluate_transition(lhs_init, rhs_init, lhs_inbox_invert, lhs_outbox_invert, rhs_inbox_invert, rhs_outbox_invert, true)
   else
   end
   

end
end
end
end
end
end
   
