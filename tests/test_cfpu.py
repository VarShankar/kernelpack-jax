import numpy as np
import jax
import jax.numpy as jnp
import pytest
from kernelpack.geometry import EmbeddedSurface,PiecewiseSmoothEmbeddedSurface,ParametricSurfaceData
from kernelpack.geometry.cfpu import _local_solve,CFPUOptions
from kernelpack.nodes import DomainNodeGenerator


def square(samples=33,h=.2,options=None):
    vertices=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1.]])
    descriptions=[]
    for a,b in zip(vertices[:-1],vertices[1:]):
        points=a+np.linspace(0,1,samples)[:,None]*(b-a)
        checks=a+np.linspace(0,1,71)[:,None]*(b-a)
        descriptions.append(ParametricSurfaceData.from_geometry(points,[points[[0,-1]]],validation_positions=checks))
    surface=PiecewiseSmoothEmbeddedSurface()
    surface.generate_piecewise_smooth_surface_by_segment(descriptions,[False]*4,h)
    surface.build_level_set(options=options)
    return surface


def test_padded_local_curl_free_fit_reproduces_plane():
    points=np.array([[-.6,-.2,0],[-.2,.4,0],[.3,-.4,0],[.6,.4,0.]])
    sites=np.zeros((1,8,3)); sites[0,:4]=points
    normals=np.zeros_like(sites); normals[0,:4,2]=1
    mask=np.zeros((1,8)); mask[0,:4]=1
    c,b,r,constant=_local_solve(jnp.asarray(sites),jnp.asarray(normals),jnp.asarray(mask))
    np.testing.assert_allclose(b,[[0,0,1]],atol=1e-10)
    np.testing.assert_allclose(c,0,atol=1e-10)
    np.testing.assert_allclose(r,0,atol=1e-10)


def test_square_interpolation_farfield_and_full_blend_gradient():
    surface=square(); level=surface.get_level_set(); assert level.mode=='cfpu'
    x=np.asarray(surface.get_bdry_nodes())
    assert np.abs(np.asarray(level.evaluate(x))).max()<1e-8
    points=np.array([[0,0],[.2,.3],[1.4,.2],[-1.4,0],[10,10.]])
    values=np.asarray(level.evaluate(points))
    assert np.all(values[:2]>0) and np.all(values[2:]<0)
    p=np.array([[.81,.13]])
    gradient=np.asarray(level.evaluate_gradient(p)); epsilon=1e-5
    finite=np.column_stack([(np.asarray(level.evaluate(p+np.eye(2)[d]*epsilon))-
        np.asarray(level.evaluate(p-np.eye(2)[d]*epsilon)))/(2*epsilon) for d in range(2)])
    np.testing.assert_allclose(gradient,finite,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(jax.jit(level.evaluate)(jnp.asarray(points)),values)
    np.testing.assert_allclose(level.evaluate_model(level.get_evaluation_model(),points),values)


def test_cube_uses_mapped_affine_face_cover():
    pytest.importorskip("warp")   # Warp Bridson sampling
    outline=np.array([[-1,-1],[1,-1],[1,1],[-1,1.]])
    surface=DomainNodeGenerator().build_piecewise_smooth_boundary_hierarchical(
        [outline]*6,[True]*6,[False]*6,[-1,1,-1,1,-1,1],[0,0,1,1,2,2],
        [True,False,True,False,True,False],.3,backend='warp')
    surface.build_level_set(); level=surface.get_level_set()
    points=np.array([[0,0,0],[.7,.3,-.2],[1.4,0,0],[0,-1.4,0],[10,10,10.]])
    values=np.asarray(level.evaluate(points))
    assert np.all(values[:2]>0) and np.all(values[2:]<0)
    assert level.cfpu_model.diagnostics['inverse_map_solves'] is False
    assert level.cfpu_model.diagnostics['patch_count']==24


def test_open_boundary_refuses_farfield_promotion():
    points=np.column_stack((np.linspace(0,1,33),np.zeros(33)))
    checks=np.column_stack((np.linspace(0,1,71),np.zeros(71)))
    data=ParametricSurfaceData.from_geometry(points,[points[[0,-1]]],validation_positions=checks)
    surface=PiecewiseSmoothEmbeddedSurface()
    surface.generate_piecewise_smooth_surface_by_segment([data],[False],.1)
    with pytest.raises(ValueError,match='closed boundary'):
        surface.build_level_set()


def test_cfpu_capacity_requests_refine_the_shared_coordinate_cover():
    surface=square(samples=65,h=.1,options=CFPUOptions(max_sites=16))
    model=surface.get_level_set().cfpu_model
    assert model.diagnostics['shared_refinement_history']
    assert max(model.diagnostics['support_counts'])<=16
    assert any(len(segment.geom_model['model'].cells)>2 for segment in surface.segments)
    assert float(surface.get_level_set().evaluate(np.array([[0.,0.]]))[0])>0
