from demos.herd_dynamics import HERD_EXPERIMENT_CASES, run_herd_tests

if __name__ == "__main__":
    run_herd_tests(
        HERD_EXPERIMENT_CASES,
        render=False,
        parallel=True,
        max_workers=15,
        generations=300,
        random_seed=12345,
    )
