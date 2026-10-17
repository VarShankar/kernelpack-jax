import numpy as np
from scipy.spatial import cKDTree
from kernelpack.geometry.cfpu import CFPUOptions, build_cfpu
from kernelpack.geometry.feature_sampling import feature_graded_cfpu_sites
from test_cfpu import square


def test_sharp_feature_enrichment_preserves_anchors_spacing_and_side_normals():
    surface=square(h=.2)
    baseline=np.asarray(surface.xb_uniform); baseline_normals=np.asarray(surface.nrmls_uniform)
    points,normals,info=feature_graded_cfpu_sites(surface,factor=2,bandwidth=2)
    assert info['added_count']>0
    np.testing.assert_array_equal(points[:len(baseline)],baseline)
    np.testing.assert_array_equal(normals[:len(baseline)],baseline_normals)
    features=np.array([[-1,-1],[-1,1],[1,-1],[1,1.]])
    extra=points[len(baseline):]; normal=normals[len(baseline):]
    distance=cKDTree(features).query(extra)[0]
    assert np.all((distance>1e-10)&(distance<.4+1e-12))
    assert cKDTree(points).query(points,k=2)[0][:,1].min()>=.1*(1-1e-10)
    # Straight incident pieces have their own outward axis normal, not an
    # average of the two normals at the nearest corner.
    np.testing.assert_allclose(np.sum(np.abs(normal)>1e-8,axis=1),1)
    assert np.all(np.sum(extra*normal,axis=1)>.999)


def test_fixed_cover_density_comparison_keeps_ambient_balls_and_nodal_zeros():
    surface=square(h=.2); original=surface.level_set.cfpu_model
    points,normals,_=feature_graded_cfpu_sites(surface)
    enriched=build_cfpu(surface.segments,points,normals,cover=original,options=CFPUOptions())
    np.testing.assert_array_equal(enriched.centers,original.centers)
    np.testing.assert_array_equal(enriched.radii,original.radii)
    assert enriched.threshold==original.threshold
    assert enriched.diagnostics['frozen_cover']
    assert np.abs(np.asarray(enriched.evaluate(points))).max()<1e-8


def test_shared_level_set_option_enriches_fit_without_changing_boundary():
    surface=square(h=.2); boundary=np.asarray(surface.xb).copy()
    surface.build_level_set(options=CFPUOptions(feature_grade_factor=2))
    assert len(surface.level_set.xd)>len(boundary)
    np.testing.assert_array_equal(surface.xb,boundary)
    assert surface.level_set.cfpu_model.diagnostics['feature_sampling']['added_count']>0
