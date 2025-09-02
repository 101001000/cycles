#pragma once


#include <portableRT/portableRT.hpp>

CCL_NAMESPACE_BEGIN

ccl_device_inline bool scene_intersect_valid(const ccl_private Ray *ray)
{
  return isfinite_safe(ray->P.x) && isfinite_safe(ray->D.x) && len_squared(ray->D) != 0.0f;
}

ccl_device_intersect bool scene_intersect(KernelGlobals kg,
    const ccl_private Ray *ray,
    const uint visibility,
    ccl_private Intersection *isect)
{

    isect->t = ray->tmax;
    isect->u = 0.0f;
    isect->v = 0.0f;
    isect->prim = PRIM_NONE;
    isect->object = OBJECT_NONE;
    isect->type = PRIMITIVE_NONE;

    if(!scene_intersect_valid(ray)) {
        return false;
    }

    prt::Ray prt_ray;
    prt_ray.origin = {ray->P.x, ray->P.y, ray->P.z};
    prt_ray.direction = {ray->D.x, ray->D.y, ray->D.z};
    prt_ray.tmin = ray->tmin;
    prt_ray.tmax = ray->tmax;
    prt_ray.self_id = ray->self.prim;
    auto hits = prt::closest_hits({prt_ray});


    if (!hits[0].valid) {
        return false;
    }

    const int id = kernel_data_fetch(object_id, hits[0].primitive_id);

    //std::cout << id << std::endl;

    isect->t = hits[0].t;
    isect->u = hits[0].u;
    isect->v = hits[0].v;
    isect->prim = hits[0].primitive_id;
    isect->type = PRIMITIVE_TRIANGLE;
    isect->object = id;

    return true;
    
}

ccl_device_intersect std::vector<bool> scene_intersect2(KernelGlobals kg,
    const std::vector<Ray> &rays,
    const std::vector<uint> visibilities,
    ccl_private std::vector<Intersection>& isects)
{
    std::vector<prt::Ray> prt_rays;
    std::vector<bool> r_hits(rays.size(), false);

    for(int i = 0; i < rays.size(); i++) {
        isects[i].t = rays[i].tmax;
        isects[i].u = 0.0f;
        isects[i].v = 0.0f;
        isects[i].prim = PRIM_NONE;
        isects[i].object = OBJECT_NONE;
        isects[i].type = PRIMITIVE_NONE;

        if(!scene_intersect_valid(&rays[i])) {
            r_hits[i] = false;
        }

        prt::Ray prt_ray;
        prt_ray.origin = {rays[i].P.x, rays[i].P.y, rays[i].P.z};
        prt_ray.direction = {rays[i].D.x, rays[i].D.y, rays[i].D.z};
        prt_ray.tmin = rays[i].tmin;
        prt_ray.tmax = rays[i].tmax;
        prt_ray.self_id = rays[i].self.prim;
        prt_rays.push_back(prt_ray);
    }

    auto hits = prt::closest_hits(prt_rays);
    
    for(int i = 0; i < hits.size(); i++) {
        r_hits[i] = hits[i].valid;
        if(hits[i].valid) {
            const int id = kernel_data_fetch(object_id, hits[i].primitive_id);
            isects[i].t = hits[i].t;
            isects[i].u = hits[i].u;
            isects[i].v = hits[i].v;
            isects[i].prim = hits[i].primitive_id;
            isects[i].type = PRIMITIVE_TRIANGLE;
            isects[i].object = id;
        }
    }
    return r_hits;
}

#ifdef __BVH_LOCAL__
template<bool single_hit = false>
ccl_device_intersect bool scene_intersect_local(KernelGlobals kg,
                                                const ccl_private Ray *ray,
                                                ccl_private LocalIntersection *local_isect,
                                                const int local_object,
                                                ccl_private uint *lcg_state,
                                                const int max_hits)
{
    std::cout << "local" << std::endl;
    return false;
}
#endif

#ifdef __VOLUME__
ccl_device_intersect bool scene_intersect_volume(KernelGlobals kg,
                                                 const ccl_private Ray *ray,
                                                 ccl_private Intersection *isect,
                                                 const uint visibility)
{
    std::cout << "volume" << std::endl;
    return false;
}
#endif

#ifdef __SHADOW_RECORD_ALL__
ccl_device_intersect bool scene_intersect_shadow_all(KernelGlobals kg,
                                                     IntegratorShadowState state,
                                                     const ccl_private Ray *ray,
                                                     const uint visibility,
                                                     const uint max_hits,
                                                     ccl_private uint *num_recorded_hits,
                                                     ccl_private float *throughput)
{
    std::cout << "shadow all" << std::endl;
    return false;
}
#endif

ccl_device_intersect bool scene_intersect_shadow(KernelGlobals kg,
    const ccl_private Ray *ray,
    const uint visibility)
{
    Intersection isect;
    return scene_intersect(kg, ray, visibility, &isect);
}

CCL_NAMESPACE_END
