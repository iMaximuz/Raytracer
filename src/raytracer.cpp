#include "vectorian.h"
#include "raytrace.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <memory>
#include <vector>

#define _USE_MATH_DEFINES

using namespace vian;

#define TOLERANCE	0.00001f

f32 RayIntersectsPlane(const Ray& ray, const Plane& plane){
	f32 result = 0;

	f32 denom = vian::Dot(plane.normal, ray.direction);
	if(abs(denom) > TOLERANCE){
		result = -plane.distance - vian::Dot(plane.normal, ray.origin) / denom;
	}
	return result;
}

f32 RayIntersectsSphere(const Ray& ray, const Sphere& sphere){
	f32 result = 0;

	vec3 sphereRelativeRayOrigin = ray.origin - sphere.position;
	f32 a = vian::Dot(ray.direction, ray.direction);
	f32 b = 2.0f* vian::Dot(ray.direction, sphereRelativeRayOrigin);
	f32 c = vian::Dot(sphereRelativeRayOrigin, sphereRelativeRayOrigin) - sphere.rad*sphere.rad;

	f32 denom = 2.0f * a;
	f32 root = sqrt(b*b - 4.0f*a*c);
	if(root > TOLERANCE){
		f32 tn = (-b - root) / denom;
		f32 tp = (-b + root) / denom;

		result = tp;
		if( tn > 0 && tn < tp){
			result = tn;
		}
	}

	return result;
}

vec3 RayCast(World* world, Ray ray, u32 rayBounces){

	vec3 finalColor = {};
	vec3 attenuation = {1.f, 1.f, 1.f};

	for(u32 rayCount = 0; rayCount < rayBounces; rayCount++){
		f32 hitDistance = F32MAX;

		u32 hitMatIndex = 0;
		vec3 hitPosition {};
		vec3 hitNormal {};

		for( u32 planeIndex = 0; planeIndex < world->planes.size(); planeIndex++){
			Plane plane = world->planes[planeIndex];
			f32 t = RayIntersectsPlane(ray, plane);
			if((t > 0) && (t < hitDistance)){
				hitDistance = t;
				hitMatIndex = plane.matIndex;
				hitPosition = ray.origin + ray.direction * t;
				hitNormal = plane.normal;
			}
		}

		for( u32 sphereIndex = 0; sphereIndex < world->spheres.size(); sphereIndex++){
			Sphere sphere  = world->spheres[sphereIndex];
			f32 t = RayIntersectsSphere(ray, sphere);
			if((t > 0) && (t < hitDistance)){
				hitDistance = t;
				hitMatIndex = sphere.matIndex;
				hitPosition = ray.origin + ray.direction * t;
				hitNormal = vian::Normalize(hitPosition - sphere.position);
			}
		}

		if(hitMatIndex > 0){
			
			Material mat = world->materials[hitMatIndex];

			finalColor = finalColor + vian::Hadamard(attenuation, mat.emissive);
			f32 angle = 1;//vian::Dot(hitNormal, -ray.direction);
			if( angle < 0)
				angle = 0;
			attenuation = vian::Hadamard(attenuation, mat.diffuse * angle);

			ray.origin = hitPosition;
			vec3 pureReflection = vian::Normalize(ray.direction - hitNormal * vian::Dot(hitNormal, ray.direction) * 2.0f);
			vec3 randomReflection = vian::Normalize(hitNormal + vec3(randomBilateral(), randomBilateral(), randomBilateral()));
			ray.direction = vian::Normalize(vian::Lerp(randomReflection, mat.scatter, pureReflection));
		}
		else{
			Material mat = world->materials[0];
			finalColor = finalColor + vian::Hadamard(attenuation, mat.emissive);
			break;
		}

	}

	return finalColor;
}


void main() {
	Image image(1280, 720);

	World world {};
	const int materialCount = 5;
	Material materials[materialCount] = {};
	materials[0].emissive = vec3(0.1f, 0.1f, 0.1f);
	materials[1].diffuse = vec3(0.6f, 0.6f, 0.6f);
	materials[1].scatter = 1.0f;
	materials[2].emissive = vec3(50.0, 0.2, 50.0);
	materials[3].emissive = vec3(0.0, 0.2, 10.0f);
	materials[3].scatter = 0.9f;
	materials[4].diffuse = vec3(1.0, 0.2, 0.0);
	materials[4].scatter = 0.7f;

	const int planeCount = 1;
	Plane planes[planeCount] = {};
	planes[0].normal = vec3(0.0f, 0.0f, 1.0f);
	planes[0].distance = 0;
	planes[0].matIndex = 1;

	const int sphereCount = 3;
	Sphere spheres[sphereCount] = {};
	spheres[0].position = vec3(0.0f, 0.0f, 0.0f);
	spheres[0].rad = 1.0f;
	spheres[0].matIndex = 4;

	spheres[1].position = vec3(1.5f, -2.0f, 2.0f);
	spheres[1].rad = 1.0f;
	spheres[1].matIndex = 1;

	spheres[2].position = vec3(-2.5f, 2.0f, 0.0f);
	spheres[2].rad = 1.0f;
	spheres[2].matIndex = 3;

	world.materials.assign(materials, materials + materialCount);
	world.planes.assign(planes, planes + planeCount);
	world.spheres.assign(spheres, spheres + sphereCount);

	vec3 cameraPosition = vec3(0, 10.0f, 1.0f);
	vec3 cameraFront = vian::Normalize(cameraPosition);
	vec3 cameraRight = vian::Normalize(vian::Cross(vec3(0, 0, 1), cameraFront));
	vec3 cameraUp = vian::Normalize(vian::Cross(cameraFront, cameraRight)); 

	f32 filmDist = 1.0f;
	f32 filmW = 1.0f;
	f32 filmH = 1.0f;

	if(image.height > image.width){
		filmW = filmH * ((f32)image.width / (f32)image.height);
	}
	else if(image.width > image.height){
		filmH = filmW * ((f32)image.height / (f32)image.width);
	}

	f32 halfFilmW = filmW * 0.5f;
	f32 halfFilmH = filmH * 0.5f;
	vec3 filmCenter = cameraPosition - (cameraFront * filmDist);

	f32 halfPixW = 0.5f / (f32)image.width;
	f32 halfPixH = 0.5f / (f32)image.height;

	if(image.pixels.size() > 0){
		
		u32* out = &image.pixels[0];
		u32 raysPerPixel = 64;
		u32 bounces = 8;

		for(u16 y = 0; y < image.height; y++){

			f32 filmY = -1.0f + (2.0f * (f32)y / (f32)image.height);

			for(u16 x = 0; x < image.width; x++){

				f32 filmX = -1.0f + (2.0f * (f32)x / (f32)image.width);
				
				f32 contribution = 1.0f / (f32)raysPerPixel;
				vec3 color = {};
				Ray ray = {};
				for(int i = 0; i < raysPerPixel; i++){
					f32 offsetX = filmX + randomBilateral() * halfPixW;
					f32 offsetY = filmY + randomBilateral() * halfPixH;

					vec3 filmPos = filmCenter + (cameraRight * halfFilmW * offsetX) + (cameraUp * halfFilmH * offsetY);

					ray.origin = cameraPosition;
					ray.direction = vian::Normalize(filmPos - cameraPosition);
					color = color + RayCast(&world, ray, bounces) * contribution;
				}

				experimental::color32 BMPColor = experimental::color32(
						experimental::LinearToRGB(color.r) * 255.0f,
						experimental::LinearToRGB(color.g) * 255.0f,
						experimental::LinearToRGB(color.b) * 255.0f,
						255.0f);

				u32 finalColor = experimental::PackColor32ToARGB(BMPColor);

				*out++ = finalColor;
				
			}
			if(!(y % 64))
				printf("\rRaycasting...%f", 100.0f * ((f32)y / (f32)image.height));
		}
		printf(" Done.\n");
		image.SaveAsBitmap("result.bmp");
	}
}


