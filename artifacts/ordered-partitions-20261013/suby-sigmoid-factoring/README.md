# Sigmoid preparation correction

The unchanged installed 5e control compiled scale in 1,054 ms and unary SiLU in 95,564 ms. Source checkpoint e9de1568 factors only sigmoid's duplicated exponential callsites, preserving branch inputs and operation order. The canonical generator propagates the same helper into 14 shaders; softplus remains unchanged.

The separate source-overlay diagnostic compiled scale in 6 ms and unary SiLU in 41,483 ms on Suby. Immediate and recorded unary SiLU wrappers produced identical words and maximum normalized error 1.2212453270876722e-15 against independent F64 equations (limit 6e-12). Both actual installed native host closures completed within 10 s; owned processes exited naturally. Public co-tenants and provider caches were not isolated, so these intervals are diagnostic observations, not a speed guarantee.

The failing-before structural test and 4,265 passing symbolic branch traces cover finite inputs, subnormals, signed zero and nonfinite operand selection; they do not substitute for physical arithmetic controls. Previous full-model 300 s preparation/10 s cleanup failure remains retained.

The ordinary candidate was packed separately from a clean committed e9de1568 export. All 1,976 members match source and installed consumer. Compared with 5e,14 shaders change plus a committed README image alt-text update; no executable JavaScript changes. The initial working-tree packaging output was rejected for provenance and the clean export repacked independently. Both archive and refreshed graph/manifest identities remain unqualified until their own physical runs. The new fixture binding retains the original F64 reference and 55-prefix contract; launchReady remains false before actual opening observations.
