import numpy as np
import jax
import jax.numpy as jnp
import pytest
from kernelpack.geometry import EmbeddedSurface,ParametricSurfaceData,PUOptions,fit_parametric_pu
from kernelpack.geometry.parametric_pu import ParametricPUModel,parameterize_surface_piece
from kernelpack.nodes import DomainNodeGenerator


def uv_grid(n):
    u,v=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n),indexing='ij')
    return np.column_stack((u.ravel(),v.ravel()))


def test_affine_reproduction_and_boundary_derivatives():
    p=uv_grid(11)
    values=np.column_stack((p[:,0]+.2*p[:,1],2*p[:,1]-.1*p[:,0],.3*p[:,0]-.2*p[:,1]+.7))
    model=fit_parametric_pu(p,values)
    checks=np.array([[-1.,-1],[0.,0],[1.,1],[-.2,.3]])
    exact=np.column_stack((checks[:,0]+.2*checks[:,1],2*checks[:,1]-.1*checks[:,0],.3*checks[:,0]-.2*checks[:,1]+.7))
    np.testing.assert_allclose(model.evaluate(checks),exact,atol=2e-11)
    jac,nr=model.frame(checks)
    np.testing.assert_allclose(jac,np.broadcast_to(np.array([[1,.2],[-.1,2],[.3,-.2]]),np.asarray(jac).shape),atol=2e-10)
    assert np.isfinite(np.asarray(nr)).all()
    assert model.diagnostics['patch_count']==4


def test_derivative_includes_pu_weight_derivatives():
    # Deliberately different affine coordinate maps on two overlapping supports.
    # Averaging their tangents alone gives the wrong derivative of the blend.
    poly=np.array([[[0,0,1],[1,0,1],[0,1,0]],[[0,0,-1],[1,0,2],[0,1,0]]],float)
    model=ParametricPUModel(jnp.zeros((2,1,2)),jnp.zeros((2,1,3)),jnp.asarray(poly),
        jnp.zeros((2,3)),jnp.array([[-.5,0],[.5,0]]),jnp.ones((2,2)),
        jnp.zeros(2),jnp.ones(2),2,[])
    point=np.array([[0.,.1]]); jac,_=model.frame(point)
    epsilon=1e-5
    finite=(np.asarray(model.evaluate(point+[[epsilon,0]]))-np.asarray(model.evaluate(point-[ [epsilon,0] ])))/(2*epsilon)
    np.testing.assert_allclose(np.asarray(jac)[:,:,0],finite,rtol=1e-8,atol=1e-8)
    assert abs(float(jac[0,2,0])-1.5)>.1


def test_curve_pu_uses_same_coordinate_fit():
    p=np.linspace(0,1,33)[:,None]; values=np.column_stack((2*p[:,0],.4*p[:,0]+1))
    model=fit_parametric_pu(p,values)
    q=np.array([[0.],[.25],[1.]])
    np.testing.assert_allclose(model.evaluate(q),np.column_stack((2*q[:,0],.4*q[:,0]+1)),atol=1e-11)
    jac,nr=model.frame(q)
    np.testing.assert_allclose(jac[:,:,0],np.tile([2,.4],(3,1)),atol=1e-10)
    assert np.isfinite(np.asarray(nr)).all()


def test_curve_descriptor_uses_shared_pu_boundary_builder():
    from kernelpack.geometry import PiecewiseSmoothEmbeddedSurface
    u=np.linspace(0,1,65)
    points=np.column_stack((2*u,.05*np.sin(np.pi*u)))
    v=np.linspace(0,1,131)
    checks=np.column_stack((2*v,.05*np.sin(np.pi*v)))
    data=ParametricSurfaceData.from_geometry(points,[points[[0,-1]]],validation_positions=checks)
    fit_only=EmbeddedSurface(); fit_only.set_data_sites(data.positions)
    fit_only.build_parametric_pu_model(data.parameters,outlines=data.outlines,
        validation_parameters=data.validation_parameters,validation_positions=checks)
    assert fit_only.get_sample_sites().shape==(2,2)
    boundary=PiecewiseSmoothEmbeddedSurface()
    boundary.generate_piecewise_smooth_surface_by_segment([data],[False],.08)
    segment=boundary.segments[0]
    assert segment.geom_model['type']=='parametric-pu'
    assert segment.geom_model['model'].diagnostics['options']==vars(PUOptions())
    np.testing.assert_allclose(np.asarray(segment.get_sample_sites())[[0,-1]],points[[0,-1]],atol=2e-6)
    assert np.isfinite(np.asarray(boundary.get_bdry_nrmls())).all()
    from scipy.spatial import cKDTree
    nodes=np.asarray(boundary.get_bdry_nodes())
    assert cKDTree(nodes).query(nodes,k=2)[0][:,1].min()>=.08*(1-5e-6)


def test_curve_projection_refuses_folded_input():
    theta=np.linspace(0,1.7*np.pi,65)
    points=np.column_stack((np.cos(theta),np.sin(theta)))
    with pytest.raises(ValueError,match='projection folds'):
        ParametricSurfaceData.from_geometry(points,[points[[0,-1]]])


def test_underresolved_model_refuses_acceptance():
    p=uv_grid(7); values=np.column_stack((p,np.sin(7*p[:,0])*np.cos(7*p[:,1])))
    q=uv_grid(12); exact=np.column_stack((q,np.sin(7*q[:,0])*np.cos(7*q[:,1])))
    with pytest.raises(ValueError,match='did not meet physical indicators'):
        fit_parametric_pu(p,values,validation_parameters=q,validation_positions=exact,
            options=PUOptions(max_refinements=0))


def test_refinement_stops_when_new_supports_lack_data():
    from kernelpack.geometry.parametric_pu import PUFitError
    p=uv_grid(7); values=np.column_stack((p,np.sin(7*p[:,0])*np.cos(7*p[:,1])))
    q=uv_grid(12); exact=np.column_stack((q,np.sin(7*q[:,0])*np.cos(7*q[:,1])))
    with pytest.raises(PUFitError,match='needs more geometry data') as caught:
        fit_parametric_pu(p,values,validation_parameters=q,validation_positions=exact)
    assert caught.value.history[0]['failed_samples']>0
    assert caught.value.history[0]['patch_count']==4


def test_pu_face_hole_is_not_refilled():
    pytest.importorskip("warp")   # Warp Bridson sampling
    outer=np.array([[-1,-1],[1,-1],[1,1],[-1,1.]])
    hole=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]])
    p=uv_grid(19); p=p[np.any(np.abs(p)>=.25,axis=1)]
    data=ParametricSurfaceData(np.column_stack((p,np.zeros(len(p)))),p,[outer,hole])
    surface=DomainNodeGenerator().build_piecewise_smooth_boundary_hierarchical(
        [data],[False],[False],[0],[-1],[False],.2,backend='warp')
    x=np.asarray(surface.get_bdry_nodes())
    assert not np.any(np.all(np.abs(x[:,:2])<.25-1e-10,axis=1))
    assert np.isfinite(np.asarray(surface.get_bdry_nrmls())).all()


def test_shared_parameterization_policy_for_plane_and_hemisphere():
    p=uv_grid(13); xyz=np.column_stack((p,np.zeros(len(p))))
    boundary=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0.]])
    plane=parameterize_surface_piece(xyz,[boundary],normals=np.tile([0,0,1.],(len(p),1)))
    assert plane.method=='plane'
    theta=np.linspace(0,2*np.pi,65,endpoint=False)
    rim=np.column_stack((np.cos(theta),np.sin(theta),np.zeros(len(theta))))
    rings=np.linspace(.15,np.pi/2,12)
    xyz=np.vstack([np.column_stack((np.sin(a)*np.cos(theta),np.sin(a)*np.sin(theta),np.full(len(theta),np.cos(a)))) for a in rings])
    radial=parameterize_surface_piece(xyz,[rim],normals=xyz)
    assert radial.method=='radial_hemisphere'
    assert np.isfinite(radial.project(xyz)).all()


def test_physical_roundoff_duplicates_are_coalesced_before_parameterization():
    p=uv_grid(9); xyz=np.column_stack((p,np.zeros(len(p))))
    boundary=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0.]])
    duplicate=xyz[7].copy(); duplicate[0]+=1e-14
    sites=np.vstack((xyz,duplicate)); normals=np.tile([0,0,1.],(len(sites),1))
    data=ParametricSurfaceData.from_geometry(sites,[boundary],normals=normals)
    assert len(data)==len(xyz)
    np.testing.assert_array_equal(data.positions,xyz)
    normals[-1]*=-1
    with pytest.raises(ValueError,match='inconsistent smooth-face normals'):
        ParametricSurfaceData.from_geometry(sites,[boundary],normals=normals)


def test_shared_builder_preserves_trimmed_curved_face_and_features():
    pytest.importorskip("warp")   # Warp Bridson sampling
    outline=np.array([[-1,-1],[1,-1],[1,-.3],[-.3,-.3],[-.3,1],[-1,1.]])
    p=uv_grid(21); p=p[(p[:,0]<=-.3+1e-12)|(p[:,1]<=-.3+1e-12)]
    values=np.column_stack((p,.1*np.sum(p*p,axis=1)))
    q=uv_grid(26); q=q[(q[:,0]<=-.3)|(q[:,1]<=-.3)]
    exact=np.column_stack((q,.1*np.sum(q*q,axis=1)))
    normals=np.column_stack((-.2*q,np.ones(len(q)))); normals/=np.linalg.norm(normals,axis=1)[:,None]
    data=ParametricSurfaceData(values,p,[outline],q,exact,normals)
    feature=np.column_stack((outline,.1*np.sum(outline*outline,axis=1)))
    gen=DomainNodeGenerator()
    surface=gen.build_piecewise_smooth_boundary_hierarchical([data],[False],[False],[0],[-1],[False],.16,
        backend='warp',feature_sites=feature,feature_normals=np.tile([0,0,1.],(len(feature),1)),feature_priorities=np.full(len(feature),2))
    x=np.asarray(surface.get_bdry_nodes())
    assert not np.any((x[:,0]>-.3+1e-9)&(x[:,1]>-.3+1e-9))
    assert np.sum(np.asarray(surface.corner_flags)==2)==len(feature)
    assert len(gen.last_info['parametric_pu_faces'])==1
    from scipy.spatial import cKDTree
    assert cKDTree(x).query(x,k=2)[0][:,1].min()>=.16*(1-5e-6)
