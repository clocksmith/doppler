import Doppler.Model

structure KernelPathRegistryEntry where
  id : String
  aliasOf : Option String
  hasFile : Bool
  deriving Repr, DecidableEq

def registryEntryShapeValid (entry : KernelPathRegistryEntry) : Bool :=
  match entry.aliasOf, entry.hasFile with
  | some _, true => false
  | none, false => false
  | _, _ => true

def findRegistryEntry? : List KernelPathRegistryEntry → String → Option KernelPathRegistryEntry
  | [], _ => none
  | entry :: rest, id =>
    if entry.id = id then some entry else findRegistryEntry? rest id

def resolveAliasRootWithFuel?
    (entries : List KernelPathRegistryEntry)
    (fuel : Nat)
    (id : String)
    (chain : List String := []) : Option String :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    if id ∈ chain then
      none
    else
      match findRegistryEntry? entries id with
      | none => none
      | some entry =>
        if !registryEntryShapeValid entry then
          none
        else
          match entry.aliasOf with
          | some target => resolveAliasRootWithFuel? entries fuel target (id :: chain)
          | none => if entry.hasFile then some entry.id else none

def resolveAliasRoot?
    (entries : List KernelPathRegistryEntry)
    (id : String)
    (chain : List String := []) : Option String :=
  resolveAliasRootWithFuel? entries (entries.length + chain.length + 1) id chain

def registryAliasAcyclic (entries : List KernelPathRegistryEntry) : Bool :=
  entries.all (fun entry => registryEntryShapeValid entry && (resolveAliasRoot? entries entry.id).isSome)

def fallbackPairValid : Dtype → Dtype → Bool
  | .f16, .f32 => true
  | .f32, .f32 => true
  | _, _ => false

theorem fallback_pair_rank_monotone
    (primary fallback : Dtype) :
    fallbackPairValid primary fallback = true →
    Dtype.rank fallback ≥ Dtype.rank primary := by
  intro hvalid
  cases primary <;> cases fallback <;> simp [fallbackPairValid, Dtype.rank] at hvalid ⊢

theorem fallback_pair_bytes_monotone
    (primary fallback : Dtype) :
    fallbackPairValid primary fallback = true →
    Dtype.bytesPerElement fallback ≥ Dtype.bytesPerElement primary := by
  intro hvalid
  cases primary <;> cases fallback <;> simp [fallbackPairValid, Dtype.bytesPerElement] at hvalid ⊢

theorem self_alias_rejected (id : String) :
    registryAliasAcyclic [{ id := id, aliasOf := some id, hasFile := false }] = false := by
  simp [registryAliasAcyclic, resolveAliasRoot?, resolveAliasRootWithFuel?, findRegistryEntry?]

theorem missing_alias_target_rejected
    (id target : String)
    (hne : target ≠ id) :
    registryAliasAcyclic [{ id := id, aliasOf := some target, hasFile := false }] = false := by
  have hne' : id ≠ target := by
    intro heq
    exact hne heq.symm
  simp [registryAliasAcyclic, resolveAliasRoot?, resolveAliasRootWithFuel?, findRegistryEntry?, hne, hne']

theorem direct_alias_to_file_resolves
    (source target : String)
    (hne : source ≠ target) :
    registryAliasAcyclic [
      { id := source, aliasOf := some target, hasFile := false },
      { id := target, aliasOf := none, hasFile := true }
    ] = true := by
  have hne' : target ≠ source := by
    intro heq
    exact hne heq.symm
  simp [registryAliasAcyclic, registryEntryShapeValid, resolveAliasRoot?, resolveAliasRootWithFuel?, findRegistryEntry?, hne, hne']

theorem alias_and_file_conflict_rejected
    (id target : String)
    (hne : target ≠ id) :
    registryAliasAcyclic [{ id := id, aliasOf := some target, hasFile := true }] = false := by
  have hne' : id ≠ target := by
    intro heq
    exact hne heq.symm
  simp [registryAliasAcyclic, registryEntryShapeValid, resolveAliasRoot?, resolveAliasRootWithFuel?, findRegistryEntry?, hne, hne']
