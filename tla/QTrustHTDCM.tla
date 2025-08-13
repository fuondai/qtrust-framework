---- MODULE QTrustHTDCM ----
EXTENDS Naturals, Sequences, TLC

CONSTANTS Shards, ValidatorsPerShard

VARIABLES trust, tau, ewmaAgg, ewmaLambda

Init == 
    /\ trust \in [1..Shards -> [1..ValidatorsPerShard -> (0..100)]]
    /\ tau \in 0..100
    /\ ewmaLambda \in 1..100 \* percent (maps to (0,1])
    /\ ewmaAgg \in [1..Shards -> [1..ValidatorsPerShard -> (0..100)]]

Eligible(s,v) == trust[s][v] >= tau

SafetyInvariant == \A s \in 1..Shards: \A v \in 1..ValidatorsPerShard: trust[s][v] < tau => ~Eligible(s,v)

\* Non-compensatory aggregation (abstract): any zero dimension forces zero aggregate
NonCompensatory(t) == \A s \in 1..Shards: \A v \in 1..ValidatorsPerShard: 
    (trust[s][v] = 0) => (ewmaAgg[s][v] = 0)

\* EWMA smoothing monotonicity bound: smoothed in [min(prev, cur), max(prev, cur)]
EWMAInvariant == \A s \in 1..Shards: \A v \in 1..ValidatorsPerShard:
    LET prev == ewmaAgg[s][v] IN
    LET cur == trust[s][v] IN
    LET a == ewmaLambda IN
    LET next == ( (100 - a) * prev + a * cur ) \div 100 IN
    next >= Min(prev, cur) /\ next <= Max(prev, cur)

Next ==
    \E s \in 1..Shards: \E v \in 1..ValidatorsPerShard:
        \E d \in -5..5:
            /\ trust' = [trust EXCEPT ![s][v] = Max(0, trust[s][v] + d)]
            /\ tau' \in tau-1 .. tau+1
            /\ ewmaAgg' = [ewmaAgg EXCEPT ![s][v] = (((100 - ewmaLambda) * ewmaAgg[s][v]) + (ewmaLambda * trust'[s][v])) \div 100]
            /\ ewmaLambda' = ewmaLambda

Spec == Init /\ [][Next]_<<trust, tau, ewmaAgg, ewmaLambda>>

THEOREM SafetyTheorem == Spec => []SafetyInvariant
THEOREM EWMABounds == Spec => []EWMAInvariant
====

