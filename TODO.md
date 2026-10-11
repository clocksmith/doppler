# Shared implementation checklist

The single canonical checklist is
[Reploid TODO.md](https://github.com/clocksmith/reploid/blob/main/TODO.md)
([workspace copy](../reploid/TODO.md)). Update that file; do not maintain a second
set of checkboxes here.

Doppler owns computation and execution state; Reploid owns coordination and
permissions; Doe is an optional compiler/runtime provider. Doppler remains
independently usable. All current statuses, dependencies and acceptance live in
the canonical checklist, not this pointer.

Computation and packaging:
[DOP-1](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-1),
[DOP-2](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-2),
[DOP-3](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-3),
[DOP-4](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-4),
[DOP-5](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-5),
[DOP-6](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-6),
[DOP-7](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-7), and
[DOP-8](https://github.com/clocksmith/reploid/blob/main/TODO.md#dop-8).

Shared handoffs: [projection consumer DD-1](https://github.com/clocksmith/reploid/blob/main/TODO.md#dd-1),
[exact application package DR-1](https://github.com/clocksmith/reploid/blob/main/TODO.md#dr-1),
[optional native executor ER-1](https://github.com/clocksmith/reploid/blob/main/TODO.md#er-1),
and [physical windows ALL-2](https://github.com/clocksmith/reploid/blob/main/TODO.md#all-2).

GOALS.md and CATSCAN.md preserve product intent and ownership. Other plans retain
contracts and historical evidence; they do not select parallel implementation.
