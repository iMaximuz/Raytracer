#include "vectorian.h"
#include <vector>
using namespace vian;


#define BMP_BM 		0x4D42

#pragma pack(push, 1)
struct Bitmap_header{
	u16 file_type;
	u32 total_size;
	u16 reserved1;
	u16 reseverd2;
	u32 pixel_offset;

	u32 info_size;
	u32 width;
	u32 height;
	u16 color_planes;
	u16 bits_per_pixel;
	u32 compression;
	u32 bitmap_size;
	u32 horizontal_resolution;
	u32 vertical_resolution;
	u32 colors;
	u32 important_colors;
};
#pragma pack(pop)

struct Image{
	u32 width;
	u32 height;
	std::vector<u32> pixels;
	Image(u32 width, u32 height){
		this->width = width;
		this->height = height;
		this->pixels.assign(width * height, 0);
	}
	~Image(){ }
	
	void SaveAsBitmap(const char* filePath){
		u32 bitmap_size = sizeof(u32) * this->width * this->height;
		
		Bitmap_header header = {};
		header.file_type = BMP_BM;
		header.total_size = sizeof(Bitmap_header) + bitmap_size;
		header.pixel_offset = sizeof(Bitmap_header);
		header.info_size = sizeof(Bitmap_header) - 14;
		header.width = this->width;
		header.height = this->height;
		header.color_planes = 1;
		header.bits_per_pixel = 32;
		header.compression = 0;
		header.bitmap_size = bitmap_size;
		header.horizontal_resolution = 0;
		header.vertical_resolution = 0;
		header.colors = 0;
		header.important_colors = 0;
	
		FILE* bitmap_file = fopen(filePath, "wb");
	
		fwrite(&header, sizeof(Bitmap_header), 1, bitmap_file);
		fwrite(&this->pixels[0], bitmap_size, 1, bitmap_file);
	
		fclose(bitmap_file);
	}
};

struct Material{
	vec3 diffuse;
	vec3 emissive;
	f32 scatter;
};

struct Ray{
    vec3 origin;
    vec3 direction;
};

struct Plane{
    vec3 normal;
    s32 distance;
    u32 matIndex;
};

struct Sphere{
    vec3 position;
    u32 rad;
    u32 matIndex;
};

struct World{
    std::vector<Plane> planes;
    std::vector<Sphere> spheres;
    std::vector<Material> materials;
};


inline f32 randomUnilateral(){
	f32 result = (f32)rand() / (f32)RAND_MAX;
	return result;
}

inline f32 randomBilateral(){
	f32 result = -1.0f + 2.0f * (randomUnilateral());;
	return result;
}