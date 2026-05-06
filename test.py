from demos.herd_dynamics import HERD_EXPERIMENT_TESTS, run_herd_tests

if __name__ == "__main__":
    run_herd_tests(
        HERD_EXPERIMENT_TESTS,
        render=False,
        parallel=True,
        max_workers=5,
        generations=300,
    )
