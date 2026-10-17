import numpy as np
import jax.numpy as jnp
from scipy.spatial import cKDTree
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.nodes.hierarchical import points_in_polygon
from kernelpack.geometry.core import _assembled_boundary_keep_mask_impl
from kernelpack.accelerators import warp_assembled_boundary_keep_mask


def test_concave_face_preserves_footprint_vertices_and_spacing():
    # L-shaped face: convex-hull filling would place nodes in the missing arm.
    outline = np.array([[0,0],[1,0],[1,.3],[.3,.3],[.3,1],[0,1.]])
    features = np.column_stack([outline, np.full(len(outline), .4)])
    gen = DomainNodeGenerator()
    surface = gen.build_piecewise_smooth_boundary_hierarchical(
        [jnp.asarray(outline)], [True], [False], [.4], [2], [False], .12,
        backend='warp', seed=7, feature_sites=features,
        feature_normals=np.tile([0,0,1.], (len(features),1)),
        feature_priorities=np.full(len(features),2))
    x = np.asarray(surface.get_bdry_nodes())
    assert not np.any((x[:,0]>.3+1e-12)&(x[:,1]>.3+1e-12))
    np.testing.assert_allclose(x[:,2], .4, atol=1e-12)
    tree = cKDTree(x)
    assert tree.query(features)[0].max() < 1e-12
    assert tree.query(x,k=2)[0][:,1].min() >= .12*(1-1e-6)
    np.testing.assert_allclose(surface.get_bdry_nrmls(), np.tile([0,0,1.], (len(x),1)))
    assert np.sum(np.asarray(surface.corner_flags)==2) == 6


def test_feature_duplicate_priority_and_ghost_direction():
    outline = np.array([[0,0],[1,0],[1,1],[0,1.]])
    # An edge candidate comes first and lies within duplicate tolerance of a
    # vertex. The vertex must survive, including its incident-normal direction.
    sites = np.array([[.01,0,0],[0,0,0.]])
    normals = np.array([[0,0,1],[-1,-1,1.]])
    surface = DomainNodeGenerator().build_piecewise_smooth_boundary_hierarchical(
        [jnp.asarray(outline)], [True], [False], [0.], [2], [False], .1,
        backend='warp', feature_sites=sites, feature_normals=normals,
        feature_priorities=np.array([1,2]))
    x = np.asarray(surface.get_bdry_nodes())
    index = np.argmin(np.linalg.norm(x,axis=1))
    np.testing.assert_allclose(x[index], 0., atol=1e-12)
    n = np.asarray(surface.get_bdry_nrmls())[index]
    np.testing.assert_allclose(n, [-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)])
    assert np.linalg.norm(x-np.array([.01,0,0]),axis=1).min() >= .01-1e-12


def test_polygon_orientation_and_boundary_inclusion():
    polygon = jnp.array([[0,0],[1,0],[1,.4],[.4,.4],[.4,1],[0,1.]])
    probes = jnp.array([[.2,.8],[.8,.2],[.8,.8],[.4,.8],[0,0],[-.1,0]])
    expected = [True,True,False,True,True,False]
    np.testing.assert_array_equal(points_in_polygon(probes,polygon),expected)
    np.testing.assert_array_equal(points_in_polygon(probes,polygon[::-1]),expected)


def test_priority_thinning_is_separated_and_covers_all_candidates():
    # Higher-priority geometry arrives later, after a nearby lower-priority
    # feature. Neither duplicate removal nor arrival order may discard it.
    points = np.array([[0,0,0],[.019,0,0],[.109,0,0],[-.09,0,0]],float)
    priority = np.array([1,3,0,0])
    for keep in (_assembled_boundary_keep_mask_impl(jnp.asarray(points),jnp.asarray(priority),.1),
                 warp_assembled_boundary_keep_mask(points,priority,.1)):
        keep = np.asarray(keep)
        assert keep[1] and not keep[0]
        retained = points[keep]
        tree = cKDTree(retained)
        assert tree.query(points)[0].max() <= .1+1e-7
        assert tree.query(retained,k=2)[0][:,1].min() >= .1-1e-7
