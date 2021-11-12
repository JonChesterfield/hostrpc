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

local inspect = require 'inspect'

local function invert(s)
   assert((s == 0) or (s == 1))
   if s == 0 then
      return 1
   else
      return 0
   end
end

local function transition(state)
   local inbox = state.inbox
   local outbox = state.outbox
   local call = 'none'
   assert ((inbox == 0) or (inbox == 1))
   assert ((outbox == 0) or (outbox == 1))
   
   io.write (string.format('[%s] passed inbox %s outbox %s', state.name, inbox, outbox))

   if inbox == 0 then
      if outbox == 0 then
         -- 00
         call = 'on_lo'
         outbox = 1
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
      end
   end

   io.write (string.format('-> %s inbox %s outbox %s\n',call, inbox, outbox))

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

local function make_transition(name, inbox_state, inbox_invert, outbox_state, outbox_invert)
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
      })

      local r = {}     
      r[inbox_state] = inbox_invert_func(t.inbox)
      r[outbox_state] = outbox_invert_func(t.outbox)
      return r
   end
   
   return f
end

local call_lhs = make_transition('L', 'mem_lhs', true, 'mem_rhs', false)
local call_rhs = make_transition('R', 'mem_rhs', false, 'mem_lhs', false)

local function call(state)
   local lhs = call_lhs(state)
   local rhs = call_rhs(state)

   -- at least one function must be a no-op for every state
   local lhs_changed = not state_equal(state, lhs)
   local rhs_changed = not state_equal(state, rhs)

   local progress = rhs_changed or lhs_changed
   local exclusion = rhs_changed ~= lhs_changed
   local lhs_fixpoint = state_equal(lhs, call_lhs(lhs))
   local rhs_fixpoint = state_equal(rhs, call_rhs(rhs))
   
   assert(progress and exclusion)
   assert(lhs_fixpoint and rhs_fixpoint)
   
   if (lhs == state) then
      return rhs
   else
      return lhs
   end
end

for i = 1, 10 do
   state = call(state)
end
