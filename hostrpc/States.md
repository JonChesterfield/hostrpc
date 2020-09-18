

  G W H T   | 
H 0   0 0   | Available
D 0 0 0     | 
            | 
    v       | 
            | 
  G W H T   | 
H 0   0 0   | Wave acquires slot
D 0 1 0     | 
            | 
    v       | 
            | 
  G W H T   | 
H 0   0 0   | Wave publishes work
D 1 1 0     | 
            | 
    v       | 
            | 
  G W H T   | 
H 1   0 0   | Host notices work
D 1 1 0     | 
            | 
    v       | 
            | 
  G W H T   | 
H 1   0 1   | Thread acquires slot
D 1 1 0     | 
            | 
    v       | 
            | 
  G W H T        G W H T         |
H 1   1 1   >  H 1   1 0         | Thread publishes work then marks itself as finished 
D 1 1 0        D 1 1 0           |
            | 
    v       | 
            | 
  G W H T   | 
H 1   1 ?   | (any wave on) device notices result
D 1 1 1     | 
            | 
    v       | 
            | 
  G W H T   | 
H 1   1 ?   | Device publishes that the slot no longer needs external attention
D 0 1 1     | 
            | 
    v       | 
            | 
  G W H T   | 
H 0   1 ?   | Host notices this change
D 0 1 1     | 
            | 
    v       | 
            | 
  G W H T   | 
H 0   1 0   | Host waits for thread to mark itself 0
D 0 1 1     | 
            | 
    v       | 
            | 
  G W H T   | 
H 0   0 0   | Host marks its slot zero
D 0 1 1     | 
            | 
    v       | 
            | 
  G W H T   | 
H 0   0 0   | Device notices
D 0 1 0     |
            | 
    v       | 
            | 
  G W H T   | 
H 0   0 0   | Wave can finally drop hold on slot and return
D 0 0 0     |







