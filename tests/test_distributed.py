# Copyright (c) 2026 AutoVision contributors. Apache-2.0.

from autovision.runtime import distributed as dist


class TestRankDetection:
    def test_unset(self):
        assert dist.get_global_rank() is None
        assert dist.get_local_rank() is None
        assert dist.get_node_rank() is None

    def test_unset_means_zero(self):
        assert dist.is_global_rank_zero()
        assert dist.is_local_rank_zero()
        assert dist.is_node_rank_zero()

    def test_global_rank_chain(self):
        env = {"RANK": "3"}
        assert dist.get_global_rank(env) == 3
        assert not dist.is_global_rank_zero(env)
        env = {"LOCAL_RANK": "1"}  # fallback when RANK absent
        assert dist.get_global_rank(env) == 1
        env = {"SLURM_PROCID": "2"}
        assert dist.get_global_rank(env) == 2
        env = {"JSM_NAMESPACE_RANK": "4"}
        assert dist.get_global_rank(env) == 4

    def test_local_rank_chain(self):
        assert dist.get_local_rank({"LOCAL_RANK": "0"}) == 0
        assert dist.get_local_rank({"SLURM_LOCALID": "5"}) == 5
        assert dist.get_local_rank({"JSM_NAMESPACE_LOCAL_RANK": "2"}) == 2

    def test_node_rank_chain(self):
        assert dist.get_node_rank({"NODE_RANK": "1"}) == 1
        assert dist.get_node_rank({"GROUP_RANK": "2"}) == 2
        assert dist.get_node_rank({"SLURM_NODEID": "0"}) == 0
        assert dist.is_node_rank_zero({"SLURM_NODEID": "0"})
        assert not dist.is_node_rank_zero({"SLURM_NODEID": "1"})

    def test_global_zero_falls_back_to_node_zero(self):
        env = {"SLURM_NODEID": "0"}  # no global rank set
        assert dist.is_global_rank_zero(env)
        env = {"SLURM_NODEID": "1"}
        assert not dist.is_global_rank_zero(env)

    def test_invalid_values_skipped(self):
        env = {"RANK": "not-a-number", "LOCAL_RANK": "7"}
        assert dist.get_global_rank(env) == 7


class TestReduceDict:
    def test_noop_without_distributed(self):
        data = {"loss": __import__("torch").tensor(1.0)}
        assert dist.reduce_dict(data) is data
