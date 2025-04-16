@jax.jit
def verify(
    x0: Array, xor: Objective, cnf: Objective, eo: Objective, nae: Objective, card: Objective, amo: Objective
) -> Array:
    @jax.jit
    def unsats_xor(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        # Even count (%2==0) of True (<0) means the XOR is UNSAT.
        unsat = jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0
        return unsat

    @jax.jit
    def unsats_cnf(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.min(assign, axis=1, where=mask, initial=float("inf")) > 0
        return unsat

    @jax.jit
    def unsats_eo(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.sum(assign < 0, axis=1, where=mask) != 1
        return unsat

    @jax.jit
    def unsats_nae(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        has_true = jnp.min(assign, axis=1, where=mask, initial=float("inf")) < 0
        has_false = jnp.max(assign, axis=1, where=mask, initial=float("-inf")) > 0
        unsat = jnp.logical_not(jnp.logical_and(has_true, has_false))
        return unsat

    @jax.jit
    def unsats_card(x0: Array, lits: Array, sign: Array, mask: Array, cards: Array):
        assign = sign * x0[lits]
        sat_count = jnp.sum(assign < 0, axis=1, where=mask)
        unsat = jnp.where(cards < 0, sat_count >= jnp.abs(cards), sat_count < cards)
        return unsat

    @jax.jit
    def unsats_amo(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.sum(assign < 0, axis=1, where=mask) > 1
        return unsat

    unsat = jnp.array(
        [
            (jnp.sum(unsats_xor(x0, xor.clauses.lits, xor.clauses.sign, xor.clauses.mask))),
            (jnp.sum(unsats_cnf(x0, cnf.clauses.lits, cnf.clauses.sign, cnf.clauses.mask))),
            (jnp.sum(unsats_eo(x0, eo.clauses.lits, eo.clauses.sign, eo.clauses.mask))),
            (jnp.sum(unsats_nae(x0, nae.clauses.lits, nae.clauses.sign, nae.clauses.mask))),
            (jnp.sum(unsats_card(x0, card.clauses.lits, card.clauses.sign, card.clauses.mask, card.cards))),
            (jnp.sum(unsats_amo(x0, amo.clauses.lits, amo.clauses.sign, amo.clauses.mask))),
        ]
    )
    return unsat

@jax.jit
def unified_verify(x0: Array, obj: Objective):
    lits = obj.clauses.lits
    sign = obj.clauses.sign
    mask = obj.clauses.mask
    cards = obj.clauses.cards
    types = obj.clauses.types
    #sparse = obj.clauses.sparse

    assign = sign * x0[lits]
    #assign = sign * jnp.einsum("v,clv->cl", x0, sparse) #sparse version.

    unsat = jnp.zeros_like(types, dtype=bool)

    # xor (unsat if an even number of true (<0) assignments)
    unsat_cond = jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0
    unsat_type = jnp.where(types == class_idno['xor'], unsat_cond, False)
    unsat = unsat | unsat_type

    # cnf (unsat if min value in assignment is false (>0))
    unsat_cond = jnp.min(assign, axis=1, where=mask, initial=jnp.inf) > 0
    unsat_type = jnp.where(types == class_idno['cnf'], unsat_cond, False)
    unsat = unsat | unsat_type

    # eo (unsat if true (<0) count != 1)
    unsat_cond = jnp.sum(assign < 0, axis=1, where=mask) != 1
    unsat_type = jnp.where(types == class_idno['eo'], unsat_cond, False)
    unsat = unsat | unsat_type

    # amo (unsat if true (<0) count >1)
    unsat_cond = jnp.sum(assign < 0, axis=1, where=mask) > 1
    unsat_type = jnp.where(types == class_idno['amo'], unsat_cond, False)
    unsat = unsat | unsat_type

    # nae (unsat IF NOT(AND(any_true, any_false))=T
    unsat_cond = jnp.logical_not(
                    jnp.logical_and(
                        (jnp.min(assign, axis=1, where=mask, initial=jnp.inf) < 0), # any true?
                        (jnp.max(assign, axis=1, where=mask, initial=-jnp.inf) > 0) # any false?
                    )
                )
    unsat_type = jnp.where(types == class_idno['nae'], unsat_cond, False)
    unsat = unsat | unsat_type

    # card
    card_count = jnp.sum(assign < 0, axis=1, where=mask)
    unsat_cond = jnp.where(cards < 0, card_count >= jnp.abs(cards), card_count < cards)
    unsat_type = jnp.where(types == class_idno['card'], unsat_cond, False)
    unsat = unsat | unsat_type

    return unsat