"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1_v, tmp2_v, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1_v, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2_v, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1_v, tmp2_v)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        
        # Constants
        zero_const = self.alloc_scratch("zero_const")
        one_const = self.alloc_scratch("one_const")
        two_const = self.alloc_scratch("two_const")
        hash_const_one_one = self.alloc_scratch("hash_const_one_one")
        hash_const_one_two = self.alloc_scratch("hash_const_one_two")
        hash_const_two_one = self.alloc_scratch("hash_const_two_one")
        hash_const_two_two = self.alloc_scratch("hash_const_two_two")
        hash_const_three_one = self.alloc_scratch("hash_const_three_one")
        hash_const_three_two = self.alloc_scratch("hash_const_three_two")
        hash_const_four_one = self.alloc_scratch("hash_const_four_one")
        hash_const_four_two = self.alloc_scratch("hash_const_four_two")
        hash_const_five_one = self.alloc_scratch("hash_const_five_one")
        hash_const_five_two = self.alloc_scratch("hash_const_five_two")
        hash_const_six_one = self.alloc_scratch("hash_const_six_one")
        hash_const_six_two = self.alloc_scratch("hash_const_six_two")
        _n_nodes = self.alloc_scratch("n_nodes")
        _forest_values_p = self.alloc_scratch("forest_values_p")
        _inp_indices_p = self.alloc_scratch("inp_indices_p")
        _inp_values_p = self.alloc_scratch("inp_values_p")

        # broadcasted version of constants
        zero_const_v = self.alloc_scratch("zero_const", VLEN)
        one_const_v = self.alloc_scratch("one_const", VLEN)
        two_const_v = self.alloc_scratch("two_const", VLEN)
        hash_const_one_one_v = self.alloc_scratch("hash_const_one_one_v", VLEN)
        hash_const_one_two_v = self.alloc_scratch("hash_const_one_two_v", VLEN)
        hash_const_two_one_v = self.alloc_scratch("hash_const_two_one_v", VLEN)
        hash_const_two_two_v = self.alloc_scratch("hash_const_two_two_v", VLEN)
        hash_const_three_one_v = self.alloc_scratch("hash_const_three_one_v", VLEN)
        hash_const_three_two_v = self.alloc_scratch("hash_const_three_two_v", VLEN)
        hash_const_four_one_v = self.alloc_scratch("hash_const_four_one_v", VLEN)
        hash_const_four_two_v = self.alloc_scratch("hash_const_four_two_v", VLEN)
        hash_const_five_one_v = self.alloc_scratch("hash_const_five_one_v", VLEN)
        hash_const_five_two_v = self.alloc_scratch("hash_const_five_two_v", VLEN)
        hash_const_six_one_v = self.alloc_scratch("hash_const_six_one_v", VLEN)
        hash_const_six_two_v = self.alloc_scratch("hash_const_six_two_v", VLEN)
        forest_values_base_v = self.alloc_scratch("forest_values_base", VLEN)
        _n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)

        self.instrs.append({
            "load": [
                ("const", _n_nodes, n_nodes),
                ("const", _forest_values_p, 7)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", _inp_indices_p, (7 + (2 ** (forest_height + 1) - 1)) ),
                ("const", _inp_values_p, (7 + (2 ** (forest_height + 1) - 1)) + batch_size )
            ] 
        })
        self.instrs.append({"load": [("const", one_const, 1), ("const", two_const, 2)]})
        self.instrs.append({
            "load": [
                ("const", hash_const_one_one, 0x7ED55D16),
                ("const", hash_const_one_two, 12),
            ], 
            "valu": [
                ("vbroadcast", zero_const_v, zero_const),
                ("vbroadcast", one_const_v, one_const),
                ("vbroadcast", two_const_v, two_const),
                ("vbroadcast", forest_values_base_v, _forest_values_p)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", hash_const_two_one, 0xC761C23C),
                ("const", hash_const_two_two, 19),
            ],
            "valu": [
                ("vbroadcast", hash_const_one_one_v, hash_const_one_one),
                ("vbroadcast", hash_const_one_two_v, hash_const_one_two)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", hash_const_three_one, 0x165667B1),
                ("const", hash_const_three_two, 5),
            ],
            "valu": [
                ("vbroadcast", hash_const_two_one_v, hash_const_two_one),
                ("vbroadcast", hash_const_two_two_v, hash_const_two_two)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", hash_const_four_one, 0xD3A2646C),
                ("const", hash_const_four_two, 9),
            ],
            "valu": [
                ("vbroadcast", hash_const_three_one_v, hash_const_three_one),
                ("vbroadcast", hash_const_three_two_v, hash_const_three_two)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", hash_const_five_one, 0xFD7046C5),
                ("const", hash_const_five_two, 3),
            ],
            "valu": [
                ("vbroadcast", hash_const_four_one_v, hash_const_four_one),
                ("vbroadcast", hash_const_four_two_v, hash_const_four_two)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", hash_const_six_one, 0xB55A4F09),
                ("const", hash_const_six_two, 16),
            ],
            "valu": [
                ("vbroadcast", hash_const_five_one_v, hash_const_five_one),
                ("vbroadcast", hash_const_five_two_v, hash_const_five_two)
            ]
        })
        self.instrs.append({
            "load": [
                ("const", zero_const, 0)
            ],
            "valu": [
                ("vbroadcast", hash_const_six_one_v, hash_const_six_one),
                ("vbroadcast", hash_const_six_two_v, hash_const_six_two),
                ("vbroadcast", _n_nodes_v, _n_nodes)
            ],
            "flow": [
                ("pause",)
            ]
        })

        self.add("debug", ("comment", "Starting loop"))
        
        num_batches = batch_size // VLEN

        iterator_constants = []
        for i in range(num_batches):
            if i == 0:
                iterator_constants.append(zero_const)
            elif (i * VLEN) == 16:
                iterator_constants.append(hash_const_six_two)
            else:
                iterator_constants.append(self.scratch_const(i * VLEN))

        address_constants = []
        for i in range(num_batches):
            address_constants.append(self.alloc_scratch(f"tmp_addr_index_{i}", 1))
            address_constants.append(self.alloc_scratch(f"tmp_addr_value_{i}", 1))
    
        for i in range(0, num_batches, 6):
            alu_slots = []

            for j in range(i, min(i + 6, num_batches)):
                alu_slots.append(("+", address_constants[2 * j], _inp_indices_p, iterator_constants[j]))
                alu_slots.append(("+", address_constants[2 * j + 1], _inp_values_p, iterator_constants[j]))
            
            self.instrs.append({"alu": alu_slots})
        
        scratch_registers = []
        pipeline_factor = 2
        for i in range(pipeline_factor):
            scratch_registers.append(self.alloc_scratch(f"tmp_idx_{i}", VLEN))
            scratch_registers.append(self.alloc_scratch(f"tmp_val_{i}", VLEN))
            scratch_registers.append(self.alloc_scratch(f"tmp_node_val_{i}", VLEN))
            scratch_registers.append(self.alloc_scratch(f"tmp_addr_forest_{i}", VLEN))
            scratch_registers.append(self.alloc_scratch(f"tmp1_v_{i}", VLEN))
            scratch_registers.append(self.alloc_scratch(f"tmp1_v_{i}", VLEN))

        for round in range(rounds):
            for i in range(0, (num_batches // pipeline_factor), 1):
                address_offset = pipeline_factor * 2 * i

                tmp_idx_1 = scratch_registers[0]
                tmp_val_1 = scratch_registers[1]
                tmp_node_val_1 = scratch_registers[2]
                tmp_addr_forest_1 = scratch_registers[3]
                tmp1_v_1 = scratch_registers[4]
                tmp2_v_1 = scratch_registers[5]

                tmp_idx_2 = scratch_registers[6]
                tmp_val_2 = scratch_registers[7]
                tmp_node_val_2 = scratch_registers[8]
                tmp_addr_forest_2 = scratch_registers[9]
                tmp1_v_2 = scratch_registers[10]
                tmp2_v_2 = scratch_registers[11]
                
                tmp_addr_index_1 = address_constants[address_offset]
                tmp_addr_value_1 = address_constants[address_offset + 1]
                
                tmp_addr_index_2 = address_constants[address_offset + 2]
                tmp_addr_value_2 = address_constants[address_offset + 3]

                # idx = mem[inp_indices_p + i]
                # val = mem[inp_values_p + i]
                self.instrs.append({"load": [("vload", tmp_idx_1, tmp_addr_index_1), ("vload", tmp_val_1, tmp_addr_value_1)]})
                # self.instrs.append({"debug": [("vcompare", tmp_idx_2, [(round, i * VLEN + k, "idx") for k in range(8)] )]})
                # self.instrs.append({"debug": [("vcompare", tmp_val_2, [(round, i * VLEN + k, "val") for k in range(8)] )]})

                # node_val = mem[forest_values_p + idx]
                self.instrs.append({
                    "valu": [("+", tmp_addr_forest_1, forest_values_base_v, tmp_idx_1)],
                    "load": [("vload", tmp_idx_2, tmp_addr_index_2), ("vload", tmp_val_2, tmp_addr_value_2)]
                })
                # self.instrs.append({"debug": [("vcompare", tmp_idx_2, [(round, (2 * i + 1) * VLEN + k, "idx") for k in range(8)] )]})
                # self.instrs.append({"debug": [("vcompare", tmp_val_2, [(round, (2 * i + 1) * VLEN + k, "val") for k in range(8)] )]})
                
                self.instrs.append({
                    "load": [("load_offset", tmp_node_val_1, tmp_addr_forest_1, 0), ("load_offset", tmp_node_val_1, tmp_addr_forest_1, 1)],
                    "valu": [("+", tmp_addr_forest_2, forest_values_base_v, tmp_idx_2)]
                })
                self.instrs.append({"load": [("load_offset", tmp_node_val_1, tmp_addr_forest_1, 2), ("load_offset", tmp_node_val_1, tmp_addr_forest_1, 3)]})
                self.instrs.append({"load": [("load_offset", tmp_node_val_1, tmp_addr_forest_1, 4), ("load_offset", tmp_node_val_1, tmp_addr_forest_1, 5)]})
                self.instrs.append({"load": [("load_offset", tmp_node_val_1, tmp_addr_forest_1, 6), ("load_offset", tmp_node_val_1, tmp_addr_forest_1, 7)]})
                # self.instrs.append({"debug":[("vcompare", tmp_node_val_1, [(round, i * VLEN + k, "node_val") for k in range(8)])]})

                # val = myhash(val ^ node_val)
                self.instrs.append({
                    "valu": [("^", tmp_val_1, tmp_val_1, tmp_node_val_1)],
                    "load": [("load_offset", tmp_node_val_2, tmp_addr_forest_2, 0), ("load_offset", tmp_node_val_2, tmp_addr_forest_2, 1)]
                })
                '''
                    HASH_STAGES = [
                        ("+", 0x7ED55D16, "+", "<<", 12),
                        ("^", 0xC761C23C, "^", ">>", 19),
                        ("+", 0x165667B1, "+", "<<", 5 ),
                        ("+", 0xD3A2646C, "^", "<<", 9 ),
                        ("+", 0xFD7046C5, "+", "<<", 3 ),
                        ("^", 0xB55A4F09, "^", ">>", 16),
                    ]
                '''
                self.instrs.append({
                    "valu": [("+", tmp1_v_1, tmp_val_1, hash_const_one_one_v), ("<<", tmp2_v_1, tmp_val_1, hash_const_one_two_v)],
                    "load": [("load_offset", tmp_node_val_2, tmp_addr_forest_2, 2), ("load_offset", tmp_node_val_2, tmp_addr_forest_2, 3)]
                })
                self.instrs.append({
                    "valu": [("+", tmp_val_1, tmp1_v_1, tmp2_v_1)],
                    "load": [("load_offset", tmp_node_val_2, tmp_addr_forest_2, 4), ("load_offset", tmp_node_val_2, tmp_addr_forest_2, 5)]
                })

                self.instrs.append({
                    "valu": [("^", tmp1_v_1, tmp_val_1, hash_const_two_one_v), (">>", tmp2_v_1, tmp_val_1, hash_const_two_two_v)],
                    "load": [("load_offset", tmp_node_val_2, tmp_addr_forest_2, 6), ("load_offset", tmp_node_val_2, tmp_addr_forest_2, 7)]
                })
                # self.instrs.append({"debug":[("vcompare", tmp_node_val_2, [(round, (2 * i + 1) * VLEN + k, "node_val") for k in range(8)])]})
                self.instrs.append({
                    "valu": [("^", tmp_val_1, tmp1_v_1, tmp2_v_1), ("^", tmp_val_2, tmp_val_2, tmp_node_val_2)]
                })

                self.instrs.append({
                    "valu": [("+", tmp1_v_1, tmp_val_1, hash_const_three_one_v), ("<<", tmp2_v_1, tmp_val_1, hash_const_three_two_v), ("+", tmp1_v_2, tmp_val_2, hash_const_one_one_v), ("<<", tmp2_v_2, tmp_val_2, hash_const_one_two_v)]
                })
                self.instrs.append({
                    "valu": [("+", tmp_val_1, tmp1_v_1, tmp2_v_1), ("+", tmp_val_2, tmp1_v_2, tmp2_v_2)]
                })

                self.instrs.append({
                    "valu": [("+", tmp1_v_1, tmp_val_1, hash_const_four_one_v), ("<<", tmp2_v_1, tmp_val_1, hash_const_four_two_v), ("^", tmp1_v_2, tmp_val_2, hash_const_two_one_v), (">>", tmp2_v_2, tmp_val_2, hash_const_two_two_v)]
                })
                self.instrs.append({
                    "valu": [("^", tmp_val_1, tmp1_v_1, tmp2_v_1), ("^", tmp_val_2, tmp1_v_2, tmp2_v_2)]
                })

                self.instrs.append({
                    "valu": [("+", tmp1_v_1, tmp_val_1, hash_const_five_one_v), ("<<", tmp2_v_1, tmp_val_1, hash_const_five_two_v), ("+", tmp1_v_2, tmp_val_2, hash_const_three_one_v), ("<<", tmp2_v_2, tmp_val_2, hash_const_three_two_v)]
                })
                self.instrs.append({
                    "valu": [("+", tmp_val_1, tmp1_v_1, tmp2_v_1), ("+", tmp_val_2, tmp1_v_2, tmp2_v_2)]
                })
                
                self.instrs.append({
                    "valu": [("^", tmp1_v_1, tmp_val_1, hash_const_six_one_v), (">>", tmp2_v_1, tmp_val_1, hash_const_six_two_v), ("+", tmp1_v_2, tmp_val_2, hash_const_four_one_v), ("<<", tmp2_v_2, tmp_val_2, hash_const_four_two_v)]
                })
                self.instrs.append({
                    "valu": [("^", tmp_val_1, tmp1_v_1, tmp2_v_1), ("^", tmp_val_2, tmp1_v_2, tmp2_v_2)]
                })
                # self.instrs.append({"debug": [("vcompare", tmp_val_1, [(round, i * VLEN + k, "hashed_val") for k in range(8)])]})

                # idx = 2 * idx + (1 if val % 2 == 0 else 2)
                self.instrs.append({
                    "valu": [("%", tmp1_v_1, tmp_val_1, two_const_v), ("*", tmp_idx_1, tmp_idx_1, two_const_v), ("+", tmp1_v_2, tmp_val_2, hash_const_five_one_v), ("<<", tmp2_v_2, tmp_val_2, hash_const_five_two_v)]
                })
                self.instrs.append({
                    "valu": [("==", tmp1_v_1, tmp1_v_1, zero_const_v), ("+", tmp_val_2, tmp1_v_2, tmp2_v_2)]
                })
                self.instrs.append({
                    "flow": [("vselect", tmp1_v_1, tmp1_v_1, one_const_v, two_const_v)],
                    "valu": [("^", tmp1_v_2, tmp_val_2, hash_const_six_one_v), (">>", tmp2_v_2, tmp_val_2, hash_const_six_two_v)]
                })
                self.instrs.append({
                    "valu": [("+", tmp_idx_1, tmp_idx_1, tmp1_v_1), ("^", tmp_val_2, tmp1_v_2, tmp2_v_2)]
                })
                # self.instrs.append({"debug": [("vcompare", tmp_idx_1, [(round, i * VLEN + k, "next_idx") for k in range(8)])]})

                # idx = 0 if idx >= n_nodes else idx
                self.instrs.append({
                    "valu": [("<", tmp1_v_1, tmp_idx_1, _n_nodes_v), ("%", tmp1_v_2, tmp_val_2, two_const_v), ("*", tmp_idx_2, tmp_idx_2, two_const_v)]
                })
                self.instrs.append({
                    "flow": [("vselect", tmp_idx_1, tmp1_v_1, tmp_idx_1, zero_const_v)],
                     "valu": [("==", tmp1_v_2, tmp1_v_2, zero_const_v)]
                })
                # self.instrs.append({"debug": [("vcompare", tmp_idx_1, [(round, i * VLEN + k, "wrapped_idx") for k in range(8)])]})

                # mem[inp_indices_p + i] = idx
                # mem[inp_values_p + i] = val
                self.instrs.append({
                    "store": [("vstore", tmp_addr_index_1, tmp_idx_1), ("vstore", tmp_addr_value_1, tmp_val_1)],
                    "flow":  [("vselect", tmp1_v_2, tmp1_v_2, one_const_v, two_const_v)]
                })
                
                self.instrs.append({
                    "valu": [("+", tmp_idx_2, tmp_idx_2, tmp1_v_2)]
                })

                self.instrs.append({
                    "valu": [("<", tmp1_v_2, tmp_idx_2, _n_nodes_v)]
                })
                self.instrs.append({
                    "flow": [("vselect", tmp_idx_2, tmp1_v_2, tmp_idx_2, zero_const_v)],
                })

                self.instrs.append({
                    "store": [("vstore", tmp_addr_index_2, tmp_idx_2), ("vstore", tmp_addr_value_2, tmp_val_2)],
                })

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        # if prints:
        # print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        # print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        # print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]
        print("CYCLES: ", machine.cycle)

    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
