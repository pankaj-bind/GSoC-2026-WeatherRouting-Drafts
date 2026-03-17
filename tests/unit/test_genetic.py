def test_bezier_curve_mutation_logic():
    """Verify Bézier mutation preserves route shape and endpoints."""
    config = Config.assign_config(Path(configpath))
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    np.random.seed(2)
    mt = RouteBlendMutation(config=config, constraints_list=constraint_list)
    X = get_dummy_route_input()
    old_route = copy.deepcopy(X)
    new_route = mt._do(None, X)

    assert old_route.shape == new_route.shape
    for i_route in range(old_route.shape[0]):
        assert np.array_equal(old_route[i_route, 0][-1, :],
                              new_route[i_route, 0][-1, :])
        assert np.array_equal(old_route[i_route, 0][0, :],
                              new_route[i_route, 0][0, :])
