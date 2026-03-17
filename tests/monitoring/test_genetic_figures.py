@pytest.mark.monitoring
def test_bezier_curve_mutation_figure(monitoring_output_dir):
    """Visualize Bézier mutation for human inspection."""
    # ... same setup as above ...
    fig, ax = graphics.generate_basemap(...)
    ax.add_collection(get_route_lc(old_route[0, 0]))
    ax.add_collection(get_route_lc(new_route[0, 0]))
    fig.savefig(os.path.join(monitoring_output_dir,
                "bezier_curve_mutation.png"), dpi=150)
    plt.close(fig)
