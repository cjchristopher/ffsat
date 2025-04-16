dummy_batch = jnp.ones((batch_size, n_vars))
dummy_weight = jnp.ones(1)

# === Basic jaxpr inspection ===
print("\n--- Basic jaxpr ---")
try:
    jaxpr = jax.make_jaxpr(process_batch)(dummy_batch, dummy_weight)
    print(jaxpr)
except Exception as e:
    print(f"Error making jaxpr: {e}")

# === Examine sharding annotations in jaxpr ===
print("\n--- Sharding annotations ---")
try:
    # Extract sharding information from the jaxpr
    sharding_info = []
    for eqn in jaxpr.jaxpr.eqns:
        if eqn.primitive.name == "with_sharding_constraint":
            params = eqn.params
            sharding_info.append(f"Variable: {eqn.invars}, Sharding: {params}")

    if sharding_info:
        print("Found sharding constraints:")
        for info in sharding_info:
            print(f"  {info}")
    else:
        print("No explicit sharding constraints found in the jaxpr")
except Exception as e:
    print(f"Error analyzing sharding: {e}")

# === Check input and output properties ===
print("\n--- I/O Properties ---")
try:
    lowered = jit_batch.lower(dummy_batch, dummy_weight)
    # Get more detailed information about the HLO module
    # hlo_module = lowered.compiler_ir()
    # print("\n--- HLO Module Info ---")
    # print(f"IR: {dir(hlo_module.context)}")
    # print(f"Module name: {hlo_module.parse()}")
    # print(f"Entry computation name: {hlo_module.entry_computation_name}")

    # # Try to extract input and output shapes from the HLO
    # try:
    #     entry = hlo_module.get_computation_by_name(hlo_module.entry_computation_name)
    #     print(f"Number of instructions: {len(entry.instructions())}")

    #     # Look for sharding annotations in HLO
    #     sharding_count = 0
    #     for instr in entry.instructions():
    #         if "sharding" in str(instr).lower():
    #             sharding_count += 1
    #     print(f"Instructions with sharding annotations: {sharding_count}")
    # except Exception as e:
    #     print(f"Could not analyze entry computation: {e}")

    # Examine the output info
    print("\n--- Output Info ---")
    out_info = lowered.out_info
    print(f"Output info: {out_info}")
except Exception as e:
    print(f"Error analyzing lowered function: {e}")

# === Check device placement and memory usage ===
print("\n--- Device and Memory Analysis ---")
print(f"Number of GPUs: {n_gpu}")
print(f"Device mesh: {mesh}")
print(f"Objective sharding spec: {obj_spec}")
print(f"Batch sharding spec: {batch_spec}")

# Try to get memory usage information from JAX directly
try:
    import gc
    gc.collect()  # Force garbage collection

    # Get current memory stats from JAX
    mem_info = jax.devices()[0].memory_stats()
    print(f"\nDevice memory stats: {mem_info}")
except Exception as e:
    print(f"Error getting memory stats: {e}")

# === Visualize sharding of a single objective ===
print("\n--- Visualize Objective Sharding ---")
try:
    with mesh:
        # Create a small example to visualize
        example_obj = objs[0]
        lits = example_obj.clauses.lits

        # Apply sharding
        clause_sharding = NamedSharding(mesh, obj_spec)
        lits_sharded = jax.device_put(lits, clause_sharding)

        # Print information about the sharded array
        print(f"Original shape: {lits.shape}")
        print(f"Sharded shape: {lits_sharded.shape}")
        print(f"Sharding spec: {lits_sharded.sharding}")

        # Show which device has which slice
        try:
            # This works in newer JAX versions
            print("Device assignment visualization:")
            if n_gpu > 1:
                # For multiple devices, check first few elements on each
                for dev_idx, dev in enumerate(devices):
                    # Try to access a small slice that should be on this device
                    # This is approximate and might not always be accurate
                    slice_size = lits.shape[0] // n_gpu
                    start_idx = dev_idx * slice_size
                    end_idx = start_idx + min(slice_size, 3)  # Just check up to 3 elements

                    sample_idxs = list(range(start_idx, min(end_idx, lits.shape[0])))
                    if sample_idxs:
                        print(f"Device {dev}: expected to have items {sample_idxs}")
            else:
                print("Only one device, all data on:", devices[0])
        except Exception as e:
            print(f"Could not visualize device assignment: {e}")
except Exception as e:
    print(f"Error during sharding visualization: {e}")
