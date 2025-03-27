@jax.jit
def verify(
    x0: Array,
    xor: ClauseArrays,
    cnf: ClauseArrays,
    eo: ClauseArrays,
    nae: ClauseArrays,
    card: tuple[ClauseArrays, Array],
    amo: ClauseArrays,
):
    @jax.jit
    def verify_xor(x0: Array, xR: Array, lits: Array, sign: Array, mask: Array):
        # assignR = sign * xR[lits]
        assign = sign * x0[lits]
        # or prod with no <0/%2, instead == -1 - stability problems?
        # unsatR = jnp.sum(jnp.sum(assignR == -1, axis=1, where=mask) % 2 == 0)
        unsat = jnp.sum(jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0)
        return unsat  # , unsatR

    @jax.jit
    def verify_cnf(x0: Array, xR: Array, lits: Array, sign: Array, mask: Array):
        # assignR = sign * xR[lits]
        assign = sign * x0[lits]
        # unsatR = jnp.sum(jnp.sum(assignR == -1, axis=1, where=mask) == 0)
        unsat = jnp.sum(jnp.sum(assign < 0, axis=1, where=mask) == 0)
        return unsat  # , unsatR

    @jax.jit
    def verify_eo(x0: Array, xR: Array, lits: Array, sign: Array, mask: Array):
        # assignR = sign * xR[lits]
        assign = sign * x0[lits]
        # unsatR = jnp.sum(jnp.sum(assignR == -1, axis=1, where=mask) != 1)
        unsat = jnp.sum(jnp.sum(assign < 0, axis=1, where=mask) != 1)
        return unsat  # , unsatR

    @jax.jit
    def verify_nae(x0: Array, xR: Array, lits: Array, sign: Array, mask: Array):
        # assignR = sign * xR[lits]
        assign = sign * x0[lits]
        # has_trueR = jnp.any(assignR == -1, axis=1, where=mask)
        # has_falseR = jnp.any(assignR == 1, axis=1, where=mask)
        # unsatR = jnp.sum(jnp.logical_not(jnp.logical_and(has_trueR, has_falseR)))
        has_true = jnp.any(assign < 0, axis=1, where=mask)
        has_false = jnp.any(assign > 0, axis=1, where=mask)
        unsat = jnp.sum(jnp.logical_not(jnp.logical_and(has_true, has_false)))
        return unsat  # , unsatR

    @jax.jit
    def verify_card(x0: Array, xR: Array, lits: Array, sign: Array, mask: Array, cards: Array):
        # assignR = sign * xR[lits]
        assign = sign * x0[lits]
        # unsatR = jnp.sum(jnp.sum(assignR == -1, axis=1, where=mask) < cards)
        unsat = jnp.sum(jnp.sum(assign < 0, axis=1, where=mask) < cards)
        return unsat  # , unsatR

    @jax.jit
    def verify_amo(x0: Array, xR: Array, lits: Array, sign: Array, mask: Array):
        # assignR = sign * xR[lits]
        assign = sign * x0[lits]
        # unsatR = jnp.sum(jnp.sum(assignR == -1, axis=1, where=mask) <= 1)
        unsat = jnp.sum(jnp.sum(assign < 0, axis=1, where=mask) <= 1)
        return unsat  # , unsatR

    xR = jnp.sign(x0)  # snap variables
    # TODO: Drop the rounded eval for now?? - non-rounded is doing the same thing anyway.
    # TODO: If a rounding scheme makes sense, it will just replace outright.
    # (v_xor, v_cnf, v_eo, v_nae, v_card, v_amo) = (verify_xor(xor), verify_cnf(cnf), verify_eo(eo), verify_nae(nae), verify_card(card), verify_amo(amo))
    # unsatR = v_xor(xor)[1] + v_cnf(cnf)[1] + v_eo(eo)[1] + v_nae(nae)[1] + v_card(card)[1] + v_amo(amo)[1]
    # unsat = v_xor(xor)[0] + v_cnf(cnf)[0] + v_eo(eo)[0] + v_nae(nae)[0] + v_card(card)[0] + v_amo(amo)[0]
    unsat = (
        verify_xor(x0, xR, *xor)
        + verify_cnf(x0, xR, *cnf)
        + verify_eo(x0, xR, *eo)
        + verify_nae(x0, xR, *nae)
        + verify_card(x0, xR, *card[0], card[-1])
        + verify_amo(x0, xR, *amo)
    )

    return unsat, None  # unsat_r


    objective_map = [
        # Distribute over objectives
        {
            "clauses": {
                "lits": 0,  # distribute lits over k
                "sign": 0,  # distribute sign over k
                "mask": 0,  # distribute mask over k
                "sparse": 0,  # distribute sparse over k
                "weight": 0,  # distribute weight over k
            },
            "ffts": {
                "dft": 0,  # distribute dft over k
                "idft": 0,  # distribute idft over k
            },
            "forward_mask": 0,  # distribute forward_mask over k
            "cards": 0,  # distribute cards over k
        },
    ]