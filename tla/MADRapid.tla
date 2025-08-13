---- MODULE MADRapid ----
EXTENDS Naturals, Sequences, TLC

CONSTANTS Shards

VARIABLES state, tau, lastBelowTau, grace

(* state: mapping from shard id to committed height *)
Init == 
    /\ state \in [1..Shards -> Nat]
    /\ tau \in 0..100
    /\ lastBelowTau \in [1..Shards -> Nat]
    /\ grace \in Nat

(* One cross-shard tx between i and j increments both if successful; never increments one without the other *)
CommitPair(i,j) == /\ i \in 1..Shards /\ j \in 1..Shards /\ i # j
                    /\ state' = [state EXCEPT ![i] = state[i] + 1, ![j] = state[j] + 1]

AbortPair(i,j) == /\ i \in 1..Shards /\ j \in 1..Shards /\ i # j
                   /\ state' = state

Next == 
    \E i \in 1..Shards: \E j \in 1..Shards: i # j /\ (CommitPair(i,j) \/ AbortPair(i,j))
    /\ lastBelowTau' = lastBelowTau
    /\ tau' = tau
    /\ grace' = grace

(* Safety: no partial commit. For any pair (i,j), commits happen in pairs; heights cannot differ by 1 due to a partial commit. *)
SafetyInvariant == \A i \in 1..Shards: \A j \in 1..Shards: ~( (state[i] - state[j]) = 1 \/ (state[j] - state[i]) = 1 )

Spec == Init /\ [][Next]_state

THEOREM SafetyTheorem == Spec => []SafetyInvariant
====

