// =======================================
// CS488/688 base code
// (written by Toshiya Hachisuka)
// =======================================
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// linear algebra 
#include "linalg.h"
using namespace linalg::aliases;
#include <Eigen/Dense>


// animated GIF writer
#include "gif.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>
#include <chrono>  // for measuring performance
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
#include <unordered_map>

// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-6f;
constexpr float RayDepthMax = 6;

bool shading = true;           
bool shadows = true;
bool imageBasedLighting = false;

// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr int globalNumParticles = 20;

bool globalEnableGravity = true;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);

bool particleGravField = false;
constexpr float gravConstant = 2e-2f;

bool particleBox = false;
bool particleConstrain = false;
float globalParticleDamping = 0.01f;

// softbody related
bool showVerticies = false;
constexpr float GlobalParticleRad = 0.005f;  // Size of default particle for collisions

bool globalMoveToMouse = false;

// dynamic camera parameters
float3 globalEye = float3(0.0f, 0.0f, 1.5f);
float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing


// mouse event
static bool mouseRightPressed;
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
int videoFrameCount = 0;
int videoId = 0;
static bool globalVidRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;

// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();



// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}

// linear interpolates between two values at t
template <typename T>
T lerp(const T& A, const T& B, const float t) {
	return t * (B - A) + A;
}

// Environment map type
enum enumEnvmType {
	ENVM_DEBEVEC,
	ENVM_USC
};
enumEnvmType globalEnvmType = ENVM_DEBEVEC;


// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	// Blurs image with low pass box filter
	// box has side length 2r + 1
	Image lowPassFilter(int r = 1) {
		assert(r >= 0);
		Image low = *this;
		int cells = (2 * r + 1) * (2 * r + 1);

		for (int i = r; i < low.width-r; i++) {
			for (int j = r; j < low.height-r; j++) {
				float3 c(0.0f);
				for (int s = -r; s <= r; s++) {
					for (int t = -r; t <= r; t++) {
						c += pixel(i + s, j + t);
					}
				}
				low.pixel(i, j) = c / cells;
			}
		}
		return low;
	}

	// Scale image by half through bilinear interpolation
	Image downSample() {
		Image down(width / 2, height / 2);

		for (int i = 0; i < down.width; i++) {
			for (int j = 0; j < down.height; j++) {
				float3 c1 = lerp(pixel(i * 2, j * 2), pixel(i * 2 + 1, j * 2), 0.5f);
				float3 c2 = lerp(pixel(i * 2, j * 2 + 1), pixel(i * 2 + 1, j * 2 + 1), 0.5f);
				down.pixel(i, j) = lerp(c1, c2, 0.5f);
			}
		}
		return down;
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	bool load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return false;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
		return true;
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		if (globalVidRecording) return;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;



// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					videoFrameCount = 0;
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_V: {
				if (!globalVidRecording) {
					videoId = int(1000.0 * PCG32::rand());
					globalVidRecording = true;
					printf("Saving in \"frames\\vid%03d\"...\n", videoId);
					printf("(Recording started)\n");
				}
				else {
					globalVidRecording = false;
					printf("(Recording done)\n");
				}
				break;
			}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			case GLFW_KEY_T: {
				if (imageBasedLighting) {
					imageBasedLighting = false;
					printf("(Ambient lighting off)\n");
				}
				else {
					imageBasedLighting = true;
					printf("(Ambient lighting on)\n");
				}
				break;
			}
			case GLFW_KEY_P: {
				// Return camera location and direction, useful for positioning
				printf("globalEye: %f %f %f\n", globalEye.x, globalEye.y, globalEye.z);
				printf("globalLookat: %f %f %f\n", globalLookat.x, globalLookat.y, globalLookat.z);
				break;
			}

			default: break;
		}
	}
}


bool grabParticle();
void releaseParticle();

// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		if (action == GLFW_PRESS) {
			if (mouseRightPressed) {
				return;
			}
			if (grabParticle()) {
				mouseRightPressed = true;
			}
		}
		else if (action == GLFW_RELEASE) {
			mouseRightPressed = false;
			releaseParticle();
		}
	}
}


// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A2, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS,
	MAT_COLGLASS   // glass with colour
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;

	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		} else if (type == MAT_LAMBERTIAN) {
			Kd = c;
		}
	}

	float3 fetchTexture(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureWidth) % textureWidth;
		int y = int(tex.y * textureHeight) % textureHeight;
		if (x < 0) x += textureWidth;
		if (y < 0) y += textureHeight;

		int pix = (x + y * textureWidth) * 3;
		const unsigned char r = texture[pix + 0];
		const unsigned char g = texture[pix + 1];
		const unsigned char b = texture[pix + 2];
		return float3(r, g, b) / 255.0f;
	}

	float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			brdfValue = Ks;
		} else if (type == MAT_GLASS) {
			// empty
		} else if (type == MAT_COLGLASS) {
				brdfValue = Kd;
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};


struct Triangle;
class TriangleMesh;

// Hit info for ray intersection
class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	float3 Ng; // geometric normal vector
	const TriangleMesh* mesh = NULL; // mesh that was hit
	const Material* material; // const pointer to the material of the intersected object
};

// Info for intersection between two objects (sphere-triangle)
class IntersectInfo {
public:
	float3 P;          // location of intersection
	float3 Ng;         // normal of triangle towards intersecting object
	const TriangleMesh* mesh = NULL; // const pointer to the mesh the triangle belongs to
	const Triangle* tri = NULL;      // pointer to the triangle that was hit
};


// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() const { return minp; };
	float3 get_maxp() const { return maxp; };
	float3 get_size() const { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}

	// PROJECT: particle collision detection
	// Returns true if sphere intersects AABB
	// Adapted from Real-Time Collision Detection (Christer Ericson, 2005), Basic Primitive Tests
	bool sphereIntersect(const float3 center, const float radius) const {

		// Compute the square distance between center and AABB
		float sqDist = 0.0f;

		for (int i = 0; i < 3; i++) {
			// For each axis count any excess distance outside box extents
			float v = center[i];
			if (v < minp[i]) sqDist += (minp[i] - v) * (minp[i] - v);
			if (v > maxp[i]) sqDist += (v - maxp[i]) * (v - maxp[i]);
		}

		// point is within sphere
		return sqDist <= radius * radius;
	}
};


class Particle;

// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float3 geoNormal;
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
	int pIndices[3] = { -1, -1, -1 }; // indexes of corresponding particles
	                                  //  (if they exist)
};

// sphere primitive for testing
struct Sphere {
	float3 center{ 0.0f };
	float radius = 1.0f;
	Material material = Material();

	bool intersect(HitInfo& result, const Ray& ray, float tMin, float tMax) const {
		float t;   // parameter for ray

		// solving quadratic equation
		float3 dist = ray.o - center;
		float A = dot(ray.d, ray.d);
		float B = 2 * dot(ray.d, dist);
		float C = dot(dist, dist) - (radius * radius);
		 
		// Getting determinant
		float discriminant = B * B - 4 * A * C;

		if (discriminant < 0) {
			return false;
		}

		t = -B / (2 * A);  // discriminant = 0
		if (discriminant > 0) {
			float t1 = (-B + sqrt(discriminant))/ (2 * A);
			float t2 = (-B - sqrt(discriminant)) / (2 * A);

			if (t1 < t2 && t1 > tMin) {
				t = t1;
			}
			else if (t2 < t1 && t2 > tMin) {
				t = t2;
			}
			else {
				return false;
			}
		}

		if (t < tMin || t > tMax) {
			return false;
		}

		result.t = t;
		result.P = ray.o + t * ray.d;
		result.N = normalize(result.P - center);
		result.Ng = result.N;
		result.T = float2(0.0f);
		result.material = &material;
		return true;
	}
};

// == Generate transform matrices ==

static float4x4 scaleMatrix(const float3 scales) {
	return float4x4{ {scales.x,0,0,0}, {0,scales.y,0,0}, {0,0,scales.z,0}, {0,0,0,1} };
}

static float4x4 translateMatrix(const float3 offset) {
	return float4x4{ {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {offset.x, offset.y,offset.z, 1} };
}

static float4x4 rotateMatrix(const float3 angles) {
	float4x4 rotateX = float4x4{ {1, 0, 0, 0},
									 {0, cosf(angles.x), sinf(angles.x), 0},
									 {0, -sinf(angles.x), cos(angles.x), 0},
									 {0,0,0,1} };
	float4x4 rotateY = float4x4{ {cosf(angles.y), 0, -sinf(angles.y), 0},
								 {0, 1, 0, 0},
								 {sinf(angles.y), 0, cos(angles.y), 0},
								 {0,0,0,1} };
	float4x4 rotateZ = float4x4{ {cosf(angles.z), sinf(angles.z), 0, 0},
								 {-sinf(angles.z), cosf(angles.z), 0, 0},
								 {0, 0, 1, 0},
								 {0,0,0,1} };
	return mul(rotateZ, mul(rotateY, rotateX));
}

float3 ClosestPtPointTriangle(float3 p, float3 a, float3 b, float3 c);
class SoftBody;

// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;
	SoftBody* body = NULL;  // corresponding softbody (if it exists)

	void transform(const float4x4& m) {
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];

				float4 P = float4(p, 1);
				float4 N = float4(n, 0);
				
				P = mul(m, P);
				N = mul(transpose(inverse(m)), N);  // transform normals by inverse of transpose

				this->triangles[i].positions[k] = float3(P.x,P.y,P.z);
				this->triangles[i].normals[k] = float3(N.x,N.y,N.z);
			}
		}
	}

	//PROJECT: Transormations
	// setting mesh transform
	void translate(const float3 offset) {
		transform(translateMatrix(offset));
	}
	void rotate(const float3 angles) {
		transform(rotateMatrix(angles));
	}
	void scale(const float3 scales) {
		transform(scaleMatrix(scales));
	}
	void setTransform(const float3 scales, const float3 angles, const float3 offset) {
		transform(mul(translateMatrix(offset), mul(rotateMatrix(angles), scaleMatrix(scales))));
	}

	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		// ====== implement it in A1 ======
		// rasterization of a triangle
		// "plm" should be a matrix that contains perspective projection and the camera matrix
		// you do not need to implement clipping
		// you may call the "shade" function to get the pixel value
		// (you may ignore viewDir for now)

		float4x3 tp;       // triangle vertices in clip space

		int pointsIn = 0;  // number of points after near plane
		bool3 in;          // which points are after near plane

		// projection to clip space
		for (int i = 0; i < 3; i++) {
			tp[i] = float4{tri.positions[i], 1.0f};
			tp[i] = mul(plm, tp[i]);  // apply perspective transform

			// point after near plane if |z| <= |w|
			in[i] = abs(tp[i].z) <= abs(tp[i].w);
			if (in[i]) {
				pointsIn += 1;
			}
		}

		// A1 Extra 2: Proper clipping
		// Clipping depending on the number of verticies after near plane

		// all points in, draw triangle as is
		if (pointsIn == 3) {
			rasterizeScreenTri(perspectiveDiv(tp), tri);
			return;
		}

		// Two points in, clip triangle at plane
		if (pointsIn == 2) {
			int iO, iI1, iI2;   // indices of two points out, one in
			for (int i = 0; i < 3; i++) {
				// find points in order p[iI1] -> p[iI2] -> p[iO]
				if (in[i] && in[(i + 1) % 3]) {
					iI1 = i;
					iI2 = (i + 1) % 3;
					iO = (i + 2) % 3;
					break;
				}
			}

			float4x4 ctp;      // points of resulting quad
			float2x4 tcoords;  // texture coordinates of resulting quad
			float t;

			// keep two in points, interpolate to get the other two at the plane
			t = nearClipLine(tp, iO, iI1);
			ctp[0] = lerp<float4>(tp[iO], tp[iI1], t);
			tcoords[0] = lerp<float2>(tri.texcoords[iO], tri.texcoords[iI1], t);

			ctp[1] = tp[iI1];
			tcoords[1] = tri.texcoords[iI1];

			ctp[2] = tp[iI2];
			tcoords[2] = tri.texcoords[iI2];

			t = nearClipLine(tp, iI2, iO);
			ctp[3] = lerp<float4>(tp[iI2], tp[iO], t);
			tcoords[3] = lerp<float2>(tri.texcoords[iI2], tri.texcoords[iO], t);

			// split quad into two triangles and rasterize separately
			float4x3 tp1 = { ctp[0], ctp[1], ctp[2] };
			float2 tcoords1[3] = { tcoords[0], tcoords[1], tcoords[2] };
			float4x3 tp2 = { ctp[0], ctp[2], ctp[3] };
			float2 tcoords2[3] = { tcoords[0], tcoords[2], tcoords[3] };

			rasterizeScreenTri(perspectiveDiv(tp1), tri);
			rasterizeScreenTri(perspectiveDiv(tp2), tri);
			return;
		}

		// One point in, clip triangle at plane
		if (pointsIn == 1) {
			int iO1, iO2, iI;   // indices of two points out, one in
			for (int i = 0; i < 3; i++) {
				// find points in order p[iO1] -> p[iO2] -> p[iI]
				if (!in[i] && !in[(i + 1) % 3]) {
					iO1 = i;
					iO2 = (i + 1) % 3;
					iI = (i + 2) % 3;
					break;
				}
			}

			// keep in point, interpolate two others at the plane
			float t;
			float2 tcoords[3];
			tcoords[iI] = tri.texcoords[iI];

			t = nearClipLine(tp, iO2, iI);
			tp[iO2] = lerp<float4>(tp[iO2], tp[iI], t);
			tcoords[iO2] = lerp<float2>(tri.texcoords[iO2], tri.texcoords[iI], t);

			t = nearClipLine(tp, iO1, iI);
			tp[iO1] = lerp<float4>(tp[iO1], tp[iI], t);
			tcoords[iO1] = lerp<float2>(tri.texcoords[iO1], tri.texcoords[iI], t);

			rasterizeScreenTri(perspectiveDiv(tp), tri);

			return;
		}
	}

	// return the t parameter for clipping between the in and out points at the near plane
	// (finds t such that z = w)
	float nearClipLine(const float4x3 &tp, const int out, const int in) const {
		float t = (tp[out].w + tp[out].z) / (tp[out].z - tp[in].z + tp[out].w - tp[in].w);
		return t;
	}

	// transforms projected points in clip space to screen space
	float4x3 perspectiveDiv(const float4x3 &tp) const {
		float4x3 ts = tp;  // triangle vertices in screen coordinates
		for (int i = 0; i < 3; i++) {
			float w = tp[i].w;
			ts[i] /= ts[i].w;            // perspective divide

			// scale [-1, 1] -> [0, globalWidth]
			ts[i].x = (ts[i].x + 1) * ((float)globalWidth / 2);
			ts[i].y = (ts[i].y + 1) * ((float)globalHeight / 2);
			ts[i].w = w;
		}
		return ts;
	}

	// Draw triangle to screen from screen coordinates
	void rasterizeScreenTri(const float4x3 &ts, const Triangle &tri) const {
		
		// A1 Task 2: Rasterize Points
		/*for (int i = 0; i < 3; i++) {
			int x = ts[i].x;
			int y = ts[i].y;

			if (FrameBuffer.valid(x, y)) {
				FrameBuffer.pixel(x, y) = float3{ 1.0f };
			}
		}
		return;*/
		
		// Calculate line test parameters
		//   such that Li(x,y) = A[i]x + B[i]y + C[i]

		// A2 Task 3: Rasterize triangle
		float3 A, B, C;
		float2 n[3];

		float3 dX, dY;
		for (int i = 0; i < 3; i++) {
			dX[i] = ts[(i + 1) % 3].x - ts[i].x;
			dY[i] = ts[(i + 1) % 3].y - ts[i].y;

			C[i] = ts[i][0] * dY[i] - ts[i][1] * dX[i];

			// get normal of line
			n[i].x = dY[i];
			n[i].y = -dX[i];
		}
		A = -1 * dY;
		B = dX;

		// draw pixels
		// area of triangle
		float area = ts[0].x * (ts[1].y - ts[2].y) + 
			         ts[1].x * (ts[2].y - ts[0].y) + 
			         ts[2].x * (ts[0].y - ts[1].y);
		float2 bc; // barycentric coordiantes of pixel

		// loop over points at center of pixels
		for (int px = 0; px < globalWidth; px++) {
			for (int py = 0; py < globalHeight; py++) {
				float x = px + 0.5;
				float y = py + 0.5;

				float3 L = x * A + y * B + C;    // Line test equations

				// pixel within triangle
				if ((L[0] >= 0 && L[1] >= 0 && L[2] >= 0) ||
					(L[0] <= 0 && L[1] <= 0 && L[2] <= 0)) {

					// top/left edge rule
					bool draw = true;
					for (int i = 0; i < 3; i++) {
						if (L[i] == 0) {
							if (n[i].x < 0 || (n[i].x == 0 && n[i].y > 0)) {
								draw = true;
								break;
							}
							else {
								draw = false;
							}
						}
					}
					if (!draw) { 
						continue; 
					}

					// interpolating z (depth)
					// compute baycentric coordinate of pixel on triangle
					bc[0] = x * (ts[1].y - ts[2].y) + y * (ts[2].x - ts[1].x) + ts[1].x * ts[2].y - ts[2].x * ts[1].y;
					bc[1] = x * (ts[2].y - ts[0].y) + y * (ts[0].x - ts[2].x) + ts[2].x * ts[0].y - ts[0].x * ts[2].y;
					bc /= area;

					float z = ts[0].z * bc[0] + ts[1].z * bc[1] + ts[2].z * (1 - bc[0] - bc[1]);

					// A1 Task 4: Depth Buffering
					// draw if point is closer than previously drawn 
					if (FrameBuffer.depth(x, y) > z) {
						HitInfo hit;
						hit.material = &materials[tri.idMaterial];

						// A1 Task 5: Perspective interpolation
						// interpolate texture coordinates
						float W = 1 / ts[0].w * bc[0] + 1 / ts[1].w * bc[1] + 1 / ts[2].w * (1 - bc[0] - bc[1]);
						hit.T = tri.texcoords[0] * bc[0] / ts[0].w + tri.texcoords[1] * bc[1] / ts[1].w + tri.texcoords[2] * (1 - bc[0] - bc[1]) / ts[2].w;
						hit.T /= W;
						hit.P = tri.positions[0] * bc[0] / ts[0].w + tri.positions[1] * bc[1] / ts[1].w + tri.positions[2] * (1 - bc[0] - bc[1]) / ts[2].w;
						hit.P /= W;
						hit.N = tri.normals[0] * bc[0] / ts[0].w + tri.normals[1] * bc[1] / ts[1].w + tri.normals[2] * (1 - bc[0] - bc[1]) / ts[2].w;
						hit.N /= W;
						hit.N = normalize(hit.N);
						hit.Ng = tri.geoNormal;
						float3 viewDir = normalize(globalEye - hit.P); // float3{ 0.0f };

						if (dot(hit.N, hit.Ng) < 0) {
							hit.Ng *= -1;
						}
						//assert(dot(hit.N, hit.Ng) > 0);
						FrameBuffer.pixel(px, py) = shade(hit, viewDir, 0);//materials[tri.idMaterial].Kd;
						FrameBuffer.depth(px, py) = z;
					}
				}
			}
		}
	}

	// A2 Task 1: Ray-triangle intersection
	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ====== implement it in A2 ======
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not

		float3 bc; // barycentric coords of intersection with triangle (alpha, beta, gamma)
		float t;   // parameter for point along ray

		// solve Mx = B where x is (Beta, Gamma, t)^T
		float3x3 M = float3x3{ {tri.positions[0].x - tri.positions[1].x, 
			                    tri.positions[0].y - tri.positions[1].y, 
			                    tri.positions[0].z - tri.positions[1].z },
							   {tri.positions[0].x - tri.positions[2].x, 
			                    tri.positions[0].y - tri.positions[2].y, 
			                    tri.positions[0].z - tri.positions[2].z },
							   {ray.d.x, ray.d.y, ray.d.z } };
		float3 B = tri.positions[0] - ray.o;

		// Getting determinant
		float D = dot(cross(M[0], M[1]), M[2]);
		
		// From Cramer's rule, get determinant for M with replaced columns
		float3x3 M_beta = M;
		float3x3 M_gamma = M;
		float3x3 M_t = M;
		M_beta[0] = B;
		M_gamma[1] = B;
		M_t[2] = B;

		float D_beta = dot(cross(M_beta[0], M_beta[1]), M_beta[2]);
		float D_gamma = dot(cross(M_gamma[0], M_gamma[1]), M_gamma[2]);
		float D_t = dot(cross(M_t[0], M_t[1]), M_t[2]);

		// test if intersecting triangle
		if ( (D > 0 && D_beta >= 0 && D_gamma >= 0 && D_beta + D_gamma <= D && tMin*D < D_t && D_t < tMax*D && D_t > 0) ||
			 (D < 0 && D_beta <= 0 && D_gamma <= 0 && D_beta + D_gamma >= D && tMin*D > D_t && D_t > tMax * D && D_t < 0)) {
			bc[1] = D_beta / D;         // beta
			bc[2] = D_gamma / D;        // gamma
			bc[0] = 1 - bc[1] - bc[2];  // alpha
			t = D_t / D;

			result.t = t;
			result.P = tri.positions[0] * bc[0] + tri.positions[1] * bc[1] + tri.positions[2] * bc[2];
			result.N = tri.normals[0] * bc[0] + tri.normals[1] * bc[1] + tri.normals[2] * bc[2];
			result.N = normalize(result.N);
			result.T = tri.texcoords[0] * bc[0] + tri.texcoords[1] * bc[1] + tri.texcoords[2] * bc[2];
			result.Ng = tri.geoNormal; //normalize(cross(tri.positions[1] - tri.positions[0], tri.positions[2] - tri.positions[0]));
			result.material = &materials[tri.idMaterial];
			result.mesh = this;

			//assert(dot(result.N, result.Ng) > 0);
			if (dot(result.N, result.Ng) < 0) {
				result.Ng *= -1;
			}

			return true;
		}
		return false;
	}

	// true if sphere intersects triangle and returns relevant intersection info
	bool sphereIntersectTri(IntersectInfo& iInfo, const Triangle& tri, const float3 center, const float radius) const {
		float3 triNormal = normalize(cross(tri.positions[1] - tri.positions[0],
			                               tri.positions[2] - tri.positions[0]));

		// distance to triangle plane
		float dist = dot(triNormal, center - tri.positions[0]);
		if (dist < 0) {
			dist *= -1;
			triNormal *= -1;
		}

		// sphere intersects plane
		if (abs(dist) <= radius) {

			// closest point on triangle to sphere
			float3 closestPoint = ClosestPtPointTriangle(center, tri.positions[0], tri.positions[1], tri.positions[2]);
			if (length2(closestPoint - center) < radius * radius) {

				iInfo.Ng = triNormal;
				iInfo.P = center - dist * triNormal;
				
				return true;
			}
		}
		return false;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
				if (materials[i].name.compare(0, 5, "colour-glass", 0, 5) == 0) {
					materials[i].type = MAT_COLGLASS;
					materials[i].eta = 1.5f;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			// getting normal of polygon
			const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
			const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
			const float3 n = normalize(cross(e0, e1));

			// setting normals
			this->triangles[i].geoNormal = n;
			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, set with normal for a polygon
				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};



// BVH node (for A2 extra)
class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};

// ====== implement it in A2 extra ======
// fill in the missing parts
class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 3.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

	// PROJECT: bvh intersect
	// Adds a new IntersectInfo for every triangle that intersects sphere
	void sphereIntersect(std::vector<IntersectInfo>& intersections, const float3 center, const float radius) const {
		// bvh
		if (this->node[0].bbox.sphereIntersect(center, radius)) {
			sphereTraverse(intersections, center, radius, 0);
		}
	}
	// traversing using sphere intersection
	void sphereTraverse(std::vector<IntersectInfo>& intersections, float3 center, 
		                float radius, int node_id) const;

private:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}

// A2 Extra 1: Faster Ray Tracing
// 
//#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
	// ====== exntend it in A2 extra ======
	bool noSplit = false;
#ifndef SAHBVH
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index  = new int[obj_num];

	float bestCost = 0;

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;
	

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	const float parentArea = bbox.area();
	float areaL = bboxL.area();
	float areaR = bboxR.area();
	bestCost = 2 * costBBox +
		       costTri * (areaL * (bestIndex + 1) +
			              areaR * (obj_num - bestIndex - 1)) / parentArea;

	bestbboxL = bboxL;
	bestbboxR = bboxR;
#else
	// implelement SAH-BVH here
	int bestAxis, bestIndex;
	int candIndex;
	float candSplit;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index;

	noSplit = true;  // true if we decide to not split
	float bestCost =  obj_num* costTri; // cost of no split
	float splitCost;
	bestIndex = 0;
	bestAxis = 0;

	const float parentArea = bbox.area();
	const int splitNum = 4;      // number of splits we test

	for (int candAxis = 0; candAxis < 3; candAxis++) {

		// sort objects along axis
		sorted_obj_index = new int[obj_num];
		this->sortAxis(obj_index, candAxis, 0, obj_num - 1);
		for (int i = 0; i < obj_num; ++i) {
			sorted_obj_index[i] = obj_index[i];
		}

		const float splitStep = bbox.get_size()[candAxis] / (splitNum + 1);
		candIndex = -1;

		candSplit = bbox.get_minp()[candAxis];
		for (int splitIndex = 1; splitIndex <= splitNum; splitIndex++) {
			candSplit += splitStep;   // point along axis that we split

			// finding object index at split
			float triMin;
			while (true) {
				if (candIndex >= obj_num-1) {
					break;
				}
				triMin = triangleMesh->triangles[sorted_obj_index[candIndex+1]].center[candAxis];
				if (triMin >= candSplit) {
					break;
				}
				candIndex++;
			}
			assert(candIndex < obj_num);

			// compute bounding boxes for current split
			bboxL.reset();
			for (int i = 0; i <= candIndex; ++i) {
				const Triangle& tri = triangleMesh->triangles[obj_index[i]];
				bboxL.fit(tri.positions[0]);
				bboxL.fit(tri.positions[1]);
				bboxL.fit(tri.positions[2]);
			}

			bboxR.reset();
			for (int i = candIndex + 1; i < obj_num; ++i) {
				const Triangle& tri = triangleMesh->triangles[obj_index[i]];
				bboxR.fit(tri.positions[0]);
				bboxR.fit(tri.positions[1]);
				bboxR.fit(tri.positions[2]);
			}

			// compute SAH for split
			float areaL = bboxL.area();
			float areaR = bboxR.area();
			splitCost = 2 * costBBox + 
				        costTri * (areaL * (candIndex+1) +
				                   areaR * (obj_num - candIndex - 1)) / parentArea;

			// record best split
			if (bestCost > splitCost) {
				bestCost = splitCost;
				bestbboxL = bboxL;
				bestbboxR = bboxR;
				bestIndex = candIndex;
				bestAxis = candAxis;
				noSplit = false;
			}
		}

		delete[] sorted_obj_index;
	}

	// resorting along best split axis
	sorted_obj_index = new int[obj_num];
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

#endif

	if (obj_num <= 4 || noSplit) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	} else {
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;

		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	//printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	//printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}


// Add intersections from children nodes
void BVH::sphereTraverse(std::vector<IntersectInfo>& intersections, const float3 center, 
	                     const float radius, int node_id) const {

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			IntersectInfo tempInfo;
			if (triangleMesh->sphereIntersectTri(tempInfo, triangleMesh->triangles[this->node[node_id].triList[i]], center, radius)) {
				tempInfo.mesh = triangleMesh;
				tempInfo.tri = &triangleMesh->triangles[this->node[node_id].triList[i]];

				intersections.push_back(tempInfo);
			}
		}
	}
	else {
		if (this->node[this->node[node_id].idLeft].bbox.sphereIntersect(center, radius)) {
			sphereTraverse(intersections, center, radius, this->node[node_id].idLeft);
		}
		if (this->node[this->node[node_id].idRight].bbox.sphereIntersect(center, radius)) {
			sphereTraverse(intersections, center, radius, this->node[node_id].idRight);
		}
	}
}








// ====== implement it in A3 ======

// A3 Task 2: Simple Collision
// axis aligned collision box
class CollisionBox {
public:
	float3 position = float3(0.0f);
	float3 size = float3(1.0f);

	CollisionBox(float3 pos = float3(0.0f), float3 s = float3(1.0f)) {
		position = pos;
		size = s;
		minp = position - (size / 2.0f);
		maxp = position + (size / 2.0f);
	}

	void setSize(float3 pos, float3 s) {
		position = pos;
		size = s;
		minp = position - (size / 2.0f);
		maxp = position + (size / 2.0f);
	}

	// If point has collided with bounding box (is on the outside)
	// returns distance to box and on which axis
	bool collide(float3& offset, float3 p) const {
		offset = float3(0.0f);
		for (int i = 0; i < 3; i++) {
			float dist = minp[i] - p[i];
			if (dist > 0) {
				offset[i] = dist;
				return true;
			}
		}
		for (int i = 0; i < 3; i++) {
			float dist = maxp[i] - p[i];
			if (dist < 0) {
				offset[i] = dist;
				return true;
			}
		}
		return false;
	}
private:
	float3 minp;
	float3 maxp;
};
CollisionBox globalColBox(float3(0.0f,0.0f,0.0f), float3(1.0f, 1.0f, 1.0f));

bool moveCollideMesh(float3& offset, float3 position, float3 prevPosition, const TriangleMesh* ignoreMesh);
class Particle {
public:
	float3 position = float3(0.0f);
	float3 prevPosition = position;

	float3 gravForce = float3(0.0f);
	float3 springForce = float3(0.0f);

	float mass = 1.0f;
	float radius = 0.0f;
	float damping = globalParticleDamping;

	void reset(float3 p = float3(FLT_MAX), float3 v = float3(FLT_MAX)) {
		if (p == float3(FLT_MAX)) {
			position = (float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f)) / 4.0f;
		}
		else { position = p; }

		float3 velocity = float3(0.0f);
		if (v == float3(FLT_MAX)) {
			velocity = 2.0f * float3((PCG32::rand() - 0.5f), 0.0f, (PCG32::rand() - 0.5f));
		}
		else { velocity = v; }
		
		prevPosition = position;
		prevPosition -= velocity * deltaT;

		if (particleConstrain) {
			constrainToSphere();
		}
	}

	void step(float timeStep = deltaT) {
		// === fill in this part in A3 ===

		// update force
		float3 force = float3(0.0f);
		if (particleGravField) {
			force += gravForce;
		}
		force += springForce;

		// update acceleration
		float3 acc = force / mass;
		// A3 Task 1: Time integration and gravity
		if (globalEnableGravity) {
			acc += globalGravity;
		}

		// update the particle position with verlet integration
		float3 temp = position;

		// verlet integration with damping
		//  (adjust damping according to the size of the sub timestep)
		float3 newpos = position + (position - prevPosition) * (1 - damping/(deltaT/timeStep)) + timeStep * timeStep * acc;

		// limit max movement to radius size in a single step (for collision)
		if (radius > 0.0f && length2(newpos - position) > radius * radius) {
			newpos = normalize(newpos - position) * radius + position;
		}
		position = newpos;
		prevPosition = temp;
	}

	void constrainToSphere(float3 center = float3(0.0f), float radius = 0.5) {
		position = center + radius * normalize(position - center);
	}

	void collideBox() {
		float3 offset;
		// if collide, get direction to box
		if (globalColBox.collide(offset, position)) {
			resolveCollision(offset);
		}
	}

	// offset is vector from position to surface of collding object
	//   assumes particle is moving into surface
	void resolveStaticCollision(const float3& offset) {

		float3 offsetN = normalize(offset);
		assert(dot(offsetN, prevPosition - position) > 0);
		float3 prevOffset = dot(offsetN, prevPosition - position) * offsetN;

		float3 oldPrev = prevPosition;
		prevPosition -= 2 * prevOffset;

		assert(dot(offsetN, position - prevPosition) > 0);

		prevPosition += offset;
		position += offset;
	}

	// offset is adjustment of position that will resolve collision
	void resolveCollision(const float3& offset) {

		float3 offsetN = normalize(offset);

		// if moving into object, flip velocity
		if (dot(offsetN, prevPosition - position) > 0) {
			float3 prevOffset = dot(offsetN, prevPosition - position) * offsetN;

			float3 oldPrev = prevPosition;
			prevPosition -= 2 * prevOffset;
		}
		prevPosition += offset;
		position += offset;
	}
};

class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	const char* sphereMeshFilePath = 0;
	float sphereSize = 0.0f;
	ParticleSystem() {};

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[0] = particles[i].position;
				particlesMesh.triangles[i].positions[1] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath)) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}

		updateMesh();
	}

	void step() {
		for (int i = 0; i < globalNumParticles; i++) {

			// A3 Task 4: Gravitational field
			if (particleGravField) {
				particles[i].gravForce = float3(0.0f);
				for (int j = 0; j < globalNumParticles; j++) {
					if (j == i) {
						continue;
					}
					particles[i].gravForce += gravityField(i, j);
				}
			}

			particles[i].step();
		}

		// A3 Task 5: Spherical particles
		// particle-particle collisions
		int collisionChecks = 1;
		if (sphereSize > 0.0f) collisionChecks = 15;

		for (int n = 0; n < collisionChecks; n++) {
			// for all pairs of particles
			for (int i = 0; i < globalNumParticles; i++) {

				if (sphereSize > 0.0f) {
					for (int j = i + 1; j < globalNumParticles; j++) {
						if (j == i) {
							continue;
						}
						if (particleCollide(i, j)) {
							resolveCollision(i, j);
						}
					}
				}

				if (particleBox) particles[i].collideBox();
				if (particleConstrain) particles[i].constrainToSphere();
			}
		}

		updateMesh();
	}

	bool particleCollide(int i, int j) {
		return length(particles[i].position - particles[j].position) <= 2 * sphereSize;
	}
	// reposition particles i and j, and set velocity according to laws of conservation
	void resolveCollision(int i, int j) {
		Particle& pi = particles[i];
		Particle& pj = particles[j];

		//initial velocities of i and j
		const float3 u_i = (pi.position - pi.prevPosition) / deltaT;
		const float3 u_j = (pj.position - pj.prevPosition) / deltaT;

		// normal of collision
		const float3 k = normalize(pi.position - pj.position);
		const float a = 2 * dot(k, u_i - u_j) / ((1 / pi.mass) + (1 / pj.mass));

		// length of intersection
		const float d = 2 * sphereSize - length(pi.position - pj.position);
		// move apart
		float3 newPos_i = pi.position + (d / 2) * k;
		float3 newPos_j = pj.position - (d / 2) * k;
	
		// new velocities
		float3 v_i = u_i - (a / particles[i].mass) * k;
		float3 v_j = u_j + (a / particles[i].mass) * k;

		// new previous position from new velocity
		particles[i].prevPosition = newPos_i - deltaT * v_i;
		particles[j].prevPosition = newPos_j - deltaT * v_j;

		particles[i].position = newPos_i;
		particles[j].position = newPos_j;

		assert(length(u_i + u_j - v_i - v_j) <= Epsilon);
	}

	float3 gravityField(int i, int j) {
		float3 disp = particles[j].position - particles[i].position;
		return gravConstant * particles[i].mass * particles[j].mass *
			disp / powf(length(disp) + gravConstant, 3);
		// constant added to avoid divide by zero
	}
};
static ParticleSystem globalParticleSystem;


// A2 Task 5: Environment mapping
//
// Returns value of environment map from direction vector
float3 ibl(Image* envMap, float3 viewDir) {
	if (envMap->width == 0 || envMap->height == 0) return float3(0.0f);

	int i, j;

	if (globalEnvmType == ENVM_DEBEVEC) {
		float r;
		viewDir = normalize(viewDir);
		if (viewDir.x == 0.0f && viewDir.y == 0.0f) {
			r = 0.0f;
		}
		else {
			r = (1 / PI) * acosf(viewDir.z) / sqrtf(powf(viewDir.x, 2) + powf(viewDir.y, 2));
		}
		i = (int)((viewDir.x * r + 1) / 2 * envMap->width) % envMap->width;
		j = (int)((viewDir.y * r + 1) / 2 * envMap->height) % envMap->height;

		if (envMap->valid(i, j)) {
			return envMap->pixel(i, j);
		}
	}
	else {
		// different format of map from USC
		float u, v;
		u = 1 + atan2f(viewDir.x, -viewDir.z) / PI;
		v = acosf(viewDir.y) / PI;

		i = (int)((u / 2) * envMap->width);
		j = (int)((1 - v) * envMap->height);

		if (envMap->valid(i, j)) {
			float3 c;
			c.x = Image::gammaCorrection(Image::toneMapping(envMap->pixel(i, j).x), 2.2);
			c.y = Image::gammaCorrection(Image::toneMapping(envMap->pixel(i, j).y), 2.2);
			c.z = Image::gammaCorrection(Image::toneMapping(envMap->pixel(i, j).z), 2.2);
			return c;
		}
	}
	
	return float3(0.0f);
}

// scene definition
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<Sphere*> debugSpheres;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<BVH> bvhs;
	Image* envMap = NULL;       // Environment map
	Image* envMapBlur = NULL;   // Blurred environment map for image-based lighting

	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc(bool recalculate = false) {
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			if (!recalculate || objects[i]->body) {
				objects[i]->preCalc();
				bvhs[i].build(objects[i]);
			}
		}
	}

	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX, 
		           bool collide = false, const TriangleMesh* ignoreMesh=NULL) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t && (!ignoreMesh || tempMinHit.mesh != ignoreMesh)) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		if (showVerticies && !collide) {
			for (int i = 0, i_n = (int)debugSpheres.size(); i < i_n; i++) {
				if (debugSpheres[i]->intersect(tempMinHit, ray, tMin, tMax)) {
					if (tempMinHit.t < minHit.t) {
						hit = true;
						minHit = tempMinHit;
					}
				}
			}
		}
		return hit;
	}

	// sphere-scene intersection, returns all triangles that intersect with the sphere
	bool sphereIntersect(std::vector<IntersectInfo>& intersections, float3 center, float radius,
	                     TriangleMesh* ignoreMesh = NULL) const {
		intersections.clear();
		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			if (objects[i] == ignoreMesh) { continue; }
			bvhs[i].sphereIntersect(intersections, center, radius);
		}
		return intersections.size();
	}

	// bruteforce sphere-scene intersection, returns all triangles that intersect with the sphere
	bool bruteSphereIntersect(std::vector<IntersectInfo>& intersections, float3 center, float radius,
		TriangleMesh* ignoreMesh = NULL) const {
		intersections.clear();
		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			if (objects[i] == ignoreMesh) { continue; }
			for (auto& tri : objects[i]->triangles) {
				IntersectInfo iInfo;
				if (objects[i]->sphereIntersectTri(iInfo, tri, center, radius)) {
					iInfo.mesh = objects[i];
					iInfo.tri = &tri;

					intersections.push_back(iInfo);
				}
			}
			bvhs[i].sphereIntersect(intersections, center, radius);
		}
		return intersections.size();
	}

	// camera -> screen matrix (given to you for A1)
	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f };

		return m;
	}

	// model -> camera matrix (given to you for A1)
	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// ====== implement it in A1 ======
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		const float4x4 plm = mul(pm, lm); // A1 Task 1: Perspective Transform matrix

		FrameBuffer.clear();
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
				objects[n]->rasterizeTriangle(objects[n]->triangles[k], plm);
			}
		}
	}

	// eye ray generation (given to you for A2)
	Ray eyeRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}

	// ray tracing (you probably don't need to change it in A2)
	void Raytrace() const {
		FrameBuffer.clear();

		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;

				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				} else if (envMap) {
					FrameBuffer.pixel(i, j) = ibl(envMap, ray.d);
				} else {
					FrameBuffer.pixel(i, j) = float3(0.0f);
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}

};
static Scene globalScene;


float3x3 sqrtM(const float3x3& M);

// related to dragging
static int grabbedParticleIndex = -1;
static SoftBody* grabbedBody = NULL;
static float tGrab = 0.3f;
float3 grabPosition();

// PROJECT: SoftBody Class
// Single softbody object implemented using particles to represent the
// physics of a loaded object, and shape matching of those particles
class SoftBody {
public:	
	// Particles that make up the object
	int numParticles = 0;
	std::vector<Particle> particles;
	std::vector<std::vector<int>> triIndices;    // the triangles associated with each particle
	std::vector<std::vector<int>> vIndices;      // the vertices associated with each particle

	const float particleRad = GlobalParticleRad; // radius of each particle (collision)
	// spheres to visualize particles (debug only)
	//   white spheres: particle position
	//   blue spheres: particle goal position
	std::vector<Sphere> spheres;

	// Starting properties of object, used to place within scene
	float3 startPosition = float3(0.0f);
	float3 startVelocity = float3(0.0f);
	float3 startRotation = float3(0.0f);  // angles for each axis
	float3 startScale = float3(1.0f);     // scales for each axis

	TriangleMesh mesh;
	const char* meshFilePath = 0;

	// control weight for tranformation vs rotation for shape matching
	float quadWeight = 0.97f;
	float springDamp = 6.0f;// 26.0f;
	float stiffness = 0.1f;

	std::vector<float3> goalPositions;

	SoftBody() {};

	// update positions of particle debug spheres
	void updateSpheres() {
		if (!showVerticies) return;
		for (int i = 0; i < numParticles; i++) {
			spheres[i * 2].center = particles[i].position;
			spheres[i * 2 + 1].center = goalPositions[i];
		}
	}

	// update the triangles that have a vertex represented by the particle
	void updateMeshVertex(int pIndex) {
		if (meshFilePath == 0) return;
		
		for (int j = 0, jn = triIndices[pIndex].size(); j < jn; j++) {
			mesh.triangles[triIndices[pIndex][j]].positions[vIndices[pIndex][j]] = particles[pIndex].position;
		}
	}

	// update all triangles according to particle positions
	void updateMeshVertices() {
		if (meshFilePath == 0) return;

		for (int i = 0; i < numParticles; i++) {
			updateMeshVertex(i);
		}
	}
	
	// PROJECT: update normals
	// calculate vertex normals as a weighted sum of neighboring face normals
	void updateMeshNormals() {
		if (meshFilePath == 0) return;

		std::vector<float3> vertexNormals(numParticles, float3(0.0f));

		// update face normals
		for (Triangle& tri : mesh.triangles) {
			tri.geoNormal = normalize(cross(tri.positions[1] - tri.positions[0], tri.positions[2] - tri.positions[0]));
		}

		// weight normals by area of face
		for (Triangle& tri : mesh.triangles) {
			// cross product is proportional area
			float3 p = cross(tri.positions[1] - tri.positions[0], tri.positions[2] - tri.positions[0]);
			vertexNormals[tri.pIndices[0]] += p;
			vertexNormals[tri.pIndices[1]] += p;
			vertexNormals[tri.pIndices[2]] += p;
		}
		for (int i = 0; i < numParticles; i++) {
			vertexNormals[i] = normalize(vertexNormals[i]);
			for (int j = 0, j_n = triIndices[i].size(); j < j_n; j++) {
				mesh.triangles[triIndices[i][j]].normals[vIndices[i][j]] = vertexNormals[i];
			}
		}
		
	}

	void initialize() {
		loadObject();

		float3 centerMassOrig = float3(0.0f);
		for (int i = 0; i < numParticles; i++) {
			goalPositions.push_back(particles[i].position);
			goalPrev.push_back(particles[i].position);

			// PROJECT: debug spheres
			// create debug spheres
			if (showVerticies) {
				// spheres at particle position
				spheres.push_back(Sphere());
				spheres[i * 2].radius = 0.006f;
				spheres[i * 2].material.type = MAT_LAMBERTIAN;
				spheres[i * 2].material.Kd = float3(1.0f, 0.9f, 0.9f);

				// spheres at particle goal position
				spheres.push_back(Sphere());
				spheres[i * 2 + 1].radius = 0.004f;
				spheres[i * 2 + 1].material.type = MAT_LAMBERTIAN;
				spheres[i * 2 + 1].material.Kd = float3(0.8, 0.8, 1);
			}

			centerMassOrig += particles[i].mass * particles[i].position;
			totalMass += particles[i].mass;
		}
		// calculate center of mass(COM) and relative positions to it
		centerMassOrig /= totalMass;
		

		// Storing original shape for shape matching
		//
		// Matrix for quadradic transformation
		AqqQ.setZero();
		for (int i = 0; i < numParticles; i++) {
			// storing original relative positions of particles to COM
			float3 q = particles[i].position - centerMassOrig;
			relativePosOrig.push_back(q);

			Eigen::Matrix<float, 9, 1> Q(
				q.x, q.y, q.z, q.x * q.x, q.y * q.y, q.z * q.z, q.x * q.y, q.y * q.z, q.z * q.x
			);
			relativePosOrigQ.push_back(Q);

			AqqQ += particles[i].mass * (Q * Q.transpose());
		}
		AqqQ = AqqQ.inverse();

		updateSpheres();
		updateMeshVertices();
		updateMeshNormals();
	}

	void step(float timeStep = deltaT) {

		// PROJECT: shape matching
		Eigen::Matrix<float, 3, 9> A;
		Eigen::Matrix<float, 3, 9> R;
		float3 centerMass;
		bool matched = shapeMatchQuad(A, R, centerMass);

		for (int i = 0; i < numParticles; i++) {
			goalPrev[i] = goalPositions[i];

			// calculate goal position for each particle

			// Project: dragging
			// grabbing overrides goal position
			if ((this == grabbedBody && i == grabbedParticleIndex) ||
				globalMoveToMouse) {
				float3 p = grabPosition();
				float3 v = p - particles[i].position;

				// limit drag distance
				if (!globalMoveToMouse && length2(v) > 0.0025f) {
					p = normalize(v) * 0.05f + particles[i].position;
				}
				goalPositions[i] = p;
			}
			else if (matched) {
				float b = quadWeight;
				if (numParticles < 9) { // quadratic transform fails for little points
					b = 0.0f;
				}

				// goal position from combination of quad tranformation and rotation
				Eigen::Vector3f g = (b * A + (1 - b) * R) * relativePosOrigQ[i];
				goalPositions[i] = float3(g(0), g(1), g(2)) + centerMass;
			}
			else {
				goalPositions[i] = particles[i].position;
			}

			// PROJECT: Spring
			// Add forces to pull particle towards goal position
			float3 d = particles[i].position - goalPositions[i];  // from goal to current
			if (length2(d) != 0.0f) {
				float3 d_n = normalize(d);

				// velocity of particle minus relative velocity between goal position
				float3 v = (particles[i].position - particles[i].prevPosition) / deltaT;
				v -= (goalPositions[i] - goalPrev[i]) / deltaT;

				// reduces velocity towards goal position
				float3 dampForce = -springDamp *  dot(v, -d_n)* (-d_n);

				// force towards goal position
				float3 pullForce = particles[i].mass* stiffness* (-d) / (timeStep * deltaT);
				particles[i].springForce = dampForce + pullForce;
			}

			particles[i].step(timeStep);
		}
		updateMeshVertices();
	}


private:
	std::vector<float3> goalPrev; // previous goal position for relative velocity

	// related to shape matching
	std::vector<float3> relativePosOrig;
	std::vector<Eigen::Matrix<float, 9, 1>> relativePosOrigQ;
	Eigen::Matrix<float, 9, 9> AqqQ;
	float totalMass = 0.0f;

	// PROJECT: load mesh
	// load mesh from obj file in meshFilePath
	void loadObject() {
		// if no mesh, load default cube
		if (meshFilePath == 0 || !mesh.load(meshFilePath)) {
			meshFilePath = 0;
			loadDefaultPoints();
			return;
		}

		// mesh loaded, transform accordingly
		mesh.body = this;
		mesh.setTransform(startScale, startRotation, startPosition);

		// Get particles from unique vertices, store corresponding triangles and vertices

		// map of vertices for Unique Position -> (Triangle Index, Vertex Index)
		std::unordered_map<float3, std::vector<std::pair<int, int>>> vertices;
		for (int i = 0; i < mesh.triangles.size(); i++) {
			for (int j = 0; j < 3; j++) {
				vertices[mesh.triangles[i].positions[j]].push_back(std::make_pair(i, j));
			}
		}

		// create particles and store triangle and vertex indices
		numParticles = 0;
		for (auto& vertex : vertices) {
			numParticles++;
			particles.push_back(Particle());
			particles.back().reset(vertex.first, startVelocity);
			particles.back().radius = particleRad;
			//particles.back().mesh = &mesh;

			triIndices.push_back({});
			vIndices.push_back({});
			for (auto& loc : vertex.second) {
				mesh.triangles[loc.first].pIndices[loc.second] = numParticles - 1;
				triIndices.back().push_back(loc.first);
				vIndices.back().push_back(loc.second);
			}
		}
	}

	// returns quadratic transformation matrix and rotation matrix for mapping current particle
	// positions to their original shape
	bool shapeMatchQuad(Eigen::Matrix<float, 3, 9>& A, Eigen::Matrix<float, 3, 9>& R, float3& centerMass) {
		if (numParticles <= 3) {
			return false;
		}

		// calculate center of mass
		centerMass = float3(0.0f);
		for (int i = 0; i < numParticles; i++) {
			centerMass += particles[i].mass * particles[i].position;
		}
		centerMass /= totalMass;

		// Matrices for quadratic and linear tranformations
		float3x3 Apq = float3x3(0.0f);
		Eigen::Matrix<float, 3, 9> ApqQ;
		ApqQ.setZero();
		for (int i = 0; i < numParticles; i++) {
			float3 relativePos = particles[i].position - centerMass;
			Apq += particles[i].mass * mul(float3x1(relativePos), transpose(relativePosOrig[i]));

			Eigen::Vector3f p(relativePos.x, relativePos.y, relativePos.z);
			ApqQ += particles[i].mass * (p * relativePosOrigQ[i].transpose());
		}
		A = ApqQ * AqqQ;  // quadratic transform matrix

		// Divide by cube root to preserve volume
		Eigen::Matrix3f a;
		a = A.block<3, 3>(0, 0);
		a /= cbrtf(a.determinant());
		A.block<3, 3>(0, 0) = a;

		// Symmetric part of rotation matrix
		float3x3 S = sqrtM(mul(transpose(Apq), Apq));
		if (determinant(S) == 0.0f) {
			//std::cout << "can't solve" << std::endl;
			return false;
		}

		// calculate rotation matrix R
		float3x3 Sinv = inverse(S);
		float3x3 r = mul(Apq, Sinv);
		const float xlength = length2(r.x);
		const float ylength = length2(r.y);
		const float zlength = length2(r.z);
		if (xlength == 0 || ylength == 0 || zlength == 0) {
			//std::cout << "can't solve 2" << std::endl;
			return false;
		}

		// pad rotation matrix to 3x9
		R.setZero();
		R(0, 0) = r.x.x;
		R(1, 0) = r.x.y;
		R(2, 0) = r.x.z;
		R(0, 1) = r.y.x;
		R(1, 1) = r.y.y;
		R(2, 1) = r.y.z;
		R(0, 2) = r.z.x;
		R(1, 2) = r.z.y;
		R(2, 2) = r.z.z;

		return true;
	}

	// load particles without mesh, shape of a cube of size s and rotated by a
	void loadDefaultPoints() {
		float s = 0.08;
		float a = PI / 4;
		std::vector<float3> points = { float3(s,-s,s),    float3(s,0.0,s),    float3(s,s,s),
									   float3(0.0,-s,s),  float3(0.0,0.0,s),  float3(0.0,s,s),
									   float3(-s,-s,s),   float3(-s,0.0,s),   float3(-s,s,s),
									   float3(-s,-s,0.0), float3(-s,0.0,0.0), float3(-s,s,0.0),
									   float3(-s,-s,-s),  float3(-s,0.0,-s),  float3(-s,s,-s),
									   float3(0.0,-s,-s), float3(0.0,0.0,-s), float3(0.0,s,-s),
									   float3(s,-s,-s),   float3(s,0.0,-s),   float3(s,s,-s),
									   float3(s,-s,0.0),  float3(s,0.0,0.0),  float3(s,s,0.0),
									   float3(0.0,s,0.0),  float3(0.0,-s,0.0) };
		float3x3 Rx = float3x3({ 1.0f, 0.0f, 0.0f }, { 0.0f, cosf(a), sinf(a) }, { 0.0f, -sinf(a), cosf(a) });
		float3x3 Rz = float3x3({ cosf(a), sinf(a), 0.0f }, { -sinf(a), cosf(a), 0 }, { 0.0f, 0, 1 });

		for (auto& p : points) {
			p = mul(Rx, mul(Rz, p));
		}
		numParticles = points.size();
		particles.resize(numParticles);
		for (int i = 0; i < numParticles; i++) {
			particles[i].reset(points[i] + startPosition, startVelocity);
			particles[i].radius = particleRad;
		}
	}
};

// Holds all physics bodies in the scene
class PhysicsBodies {
public:
	int numBodies = 0;

	const int substeps = 5;                      // substeps per frame
	const float substepSize = deltaT / substeps; // size of substep

	std::vector<SoftBody*> bodies;
	const float particleRad = GlobalParticleRad;

	// Project: Substepping
	void step() {
		for (int n = 0; n < substeps; n++) {
			substep();
		}
		updateMeshes();
	}

	void substep() {
		for (auto& body : bodies) {
			body->step(substepSize);
		}

		resolveConstraints();
	}

	void updateMeshes() {
		for (auto& body : bodies) {
			body->updateSpheres();
			body->updateMeshNormals();
		}
	}

	// resolve all collisions with collision box and meshes
	void resolveConstraints() {
		int numCollisionChecks = 15;
		for (int c = 0; c < numCollisionChecks; c++) {
			for (auto& body : bodies) {
				resolveCollisionBody(body);
			}

			if (particleBox) {
				for (auto& body : bodies) {
					for (auto& p : body->particles) {
						p.collideBox();
					}
				}
			}
		}
	}

	// resolve collisions between body and all other meshes
	void resolveCollisionBody(SoftBody* body);

	// PROJECT: particle collision response
	// resolve collision between particle and triangle
	void resolveCollision(float3& offset, SoftBody* partBody, const int partIndex, 
		                   const TriangleMesh* triMesh, const Triangle& tri) const {
		Particle& particle = partBody->particles[partIndex];

		// particle-static mesh collision
		//    reposition only particle
		if (!triMesh->body) {
			particle.resolveCollision(offset);
			partBody->updateMeshVertex(partIndex);
			return;
		}

		// particle - softbody collision
		//    reposition particle and triangle by half the offset
		particle.resolveCollision(offset/2);
		partBody->updateMeshVertex(partIndex);

		triMesh->body->particles[tri.pIndices[0]].resolveCollision(-offset / 2);
		triMesh->body->particles[tri.pIndices[1]].resolveCollision(-offset / 2);
		triMesh->body->particles[tri.pIndices[2]].resolveCollision(-offset / 2);
		triMesh->body->updateMeshVertex(tri.pIndices[0]);
		triMesh->body->updateMeshVertex(tri.pIndices[1]);
		triMesh->body->updateMeshVertex(tri.pIndices[2]);

		return;
	}

	void addBody(SoftBody* body) {
		bodies.push_back(body);
		numBodies++;
	}

	~PhysicsBodies() {
		numBodies = 0;
		bodies.clear();
	}
};
static PhysicsBodies globalPhysBodies;

// resolve collisions between body and all other meshes
void PhysicsBodies::resolveCollisionBody(SoftBody* body) {
	for (int i = 0, i_n = body->numParticles; i < i_n; i++) {
		Particle& particle = body->particles[i];

		// vector of intersections with particle
		std::vector<IntersectInfo> intersections;

		// get intersections (ignoring self intersections)
		/*if (globalScene.bruteSphereIntersect(intersections, particle.position, particleRad,
			&body->mesh)) {*/
		// PROJECT: bvh intersect
		if (globalScene.sphereIntersect(intersections, particle.position, particleRad, 
			                            &body->mesh)) {
			for (auto& iInfo : intersections) {

				float3 offset;
				float dist = length(particle.position - iInfo.P);

				if (dist != 0) {
					offset = (particleRad - dist + Epsilon) * (particle.position - iInfo.P) / dist;
				} else {
					offset = (particleRad - dot(particle.position - iInfo.P, iInfo.Ng) + Epsilon)
						      * iInfo.Ng;
				}
				assert(!isnan(offset.x));
				resolveCollision(offset, body, i, iInfo.mesh, *iInfo.tri);
			}
		}
	}
}

// Returns the closest point on a triangle to the given point
// Taken from Real-Time Collision Detection (Christer Ericson, 2005), Basic Primitive Tests
float3 ClosestPtPointTriangle(float3 p, float3 a, float3 b, float3 c) {
	// Check if P in vertex region outside A
	float3 ab = b - a;
	float3 ac = c - a;
	float3 ap = p - a;
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a; // barycentric coordinates (1,0,0)
	// Check if P in vertex region outside B
	float3 bp = p - b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) return b; // barycentric coordinates (0,1,0)
	// Check if P in edge region of AB, if so return projection of P onto AB
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		return a + v * ab; // barycentric coordinates (1-v,v,0)
	}
	// Check if P in vertex region outside C
	float3 cp = p - c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) return c; // barycentric coordinates (0,0,1)

	// Check if P in edge region of AC, if so return projection of P onto AC
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w,0,w)
	}
	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0,1-w,w)
	}
	// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab * v + ac * w; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

// Rotates a vector respect to an upward facing normal to a new normal
float3 rotateAroundNormal(float3 v, float3 n) {
	const float3 nBase = float3(0.0f, 0.0f, 1.0f);
	if (n == nBase) {
		return v;
	}
	else if (n == -nBase) {
		return -v;
	}
	float3 axis = normalize(cross(nBase, n));
	float cosAngle = dot(nBase, n) / (length(nBase) * length(n));
	float sinAngle = sinf(acosf(cosAngle));

	return v * cosAngle + cross(axis, v) * sinAngle + axis * dot(axis, v) * (1 - cosAngle);
}

// returns random cosine-weighted direction in the hemisphere defined by normal
float3 cosWeightedDir(float3 normal) {
	float r1 = PCG32::rand();
	float r2 = PCG32::rand();

	float root = sqrtf(1 - r2);

	// generate cosine-weighted direction for upward hemisphere
	float3 dir(cosf(2 * PI * r1) * root, sinf(2 * PI * r1) * root, sqrtf(r2));
	
	return rotateAroundNormal(dir, normal);  // rotate to normal
}


// ====== implement it in A2 ======
// fill in the missing parts
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level) {
	if (!shading) {
		return hit.material->Kd;
	}
	if (hit.material->type == MAT_LAMBERTIAN) {

		// flipping normal to be on viewing side
		float3 normal = hit.N;
		if (dot(viewDir, hit.Ng) < 0) {
			normal = -normal;
		}

		// you may want to add shadow ray tracing here in A2
		float3 L = float3(0.0f);
		float3 brdf, irradiance;

		// loop over all of the point light sources
		for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
			float3 l = globalScene.pointLightSources[i]->position - hit.P;

			// the inverse-squared falloff
			const float falloff = length2(l);

			// normalize the light direction
			l /= sqrtf(falloff);

			// get the irradiance
			irradiance = float(std::max(0.0f, dot(normal, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
			brdf = hit.material->BRDF(l, viewDir, normal);
			
			if (hit.material->isTextured) {
				brdf *= hit.material->fetchTexture(hit.T);
			}
			//return brdf * PI; //debug output

			// A2 Task 2: Shadow ray tracing
			// 
			// In shadow if there is an intersection between hit point and light source
			if (shadows) {
				HitInfo blockHit;
				const Ray shadowRay(hit.P, l);
				float tLight = (globalScene.pointLightSources[i]->position.x - hit.P.x) / l.x;
				if (globalScene.intersect(blockHit, shadowRay, Epsilon, tLight)) {
					continue;  // go to next light source
				}
			}

			L += irradiance * brdf;
		}

		// A2 Extra 2: Lambertian image-based lighting (Brute force approach)
		//   Send a few random rays for each point
		//   PRESS 'T' TO ACTIVATE
		if (imageBasedLighting && globalScene.envMapBlur) {
			assert(globalScene.envMapBlur->width > 0);

			// number of rays to send
			int n = 8;
			float3 Ei = float3(0.0f);
			for (int i = 0; i < n; i++) {

				// random cosine weighted direction
				float3 randDir = cosWeightedDir(normal);

				Ray randRay(hit.P, randDir);
				HitInfo randHit;
				if (!globalScene.intersect(randHit, randRay, Epsilon)) {
					Ei +=  ibl(globalScene.envMapBlur, randDir);
				}
			}
			Ei /= n; // average irradiance

			brdf = hit.material->BRDF(normal, viewDir, normal);
			if (hit.material->isTextured) {
				brdf *= hit.material->fetchTexture(hit.T);
			}
			
			L += brdf * Ei;
		}

		return L;

	} else if (hit.material->type == MAT_METAL) {

		// A2 Task 3: Specular reflection

		if (level >= RayDepthMax) {
			return float3(0.0f);
		}

		// flipping normals to be on viewing side
		float3 normal = hit.N;
		float3 geoNorm = hit.Ng;
		if (dot(viewDir, normal) < 0) {
			normal = -normal;
			geoNorm = -geoNorm;
		}
		assert(dot(viewDir, normal) > 0);

		float3 brdf;

		// ray into surface
		float3 wi = -viewDir;
		assert(dot(wi, normal) < 0);

		// reflected ray out of surface
		float3 wr = -2 * dot(wi, normal) * normal + wi;
		assert(dot(wr, normal) > 0);

		// edge case to make reflected ray consistent with geometric normal
		if (dot(geoNorm, wr) < 0) {
			wr = wr - (2 * dot(wr,geoNorm) * geoNorm);
		}
		assert(dot(geoNorm, wr) > 0);

		brdf = hit.material->BRDF(-wi, wr, normal);

		// reflected ray hits object
		Ray reflectedRay(hit.P, wr);
		HitInfo reflectedHit;
		if (globalScene.intersect(reflectedHit, reflectedRay, Epsilon)) {
			return  brdf * shade(reflectedHit, -wr, level+1);  // return shade of object
		}
		// reflected ray hits environment map
		if (globalScene.envMap) {
			return brdf * ibl(globalScene.envMap, wr);
		}

		return float3(0.0f);

	} else if (hit.material->type == MAT_GLASS ||
		       hit.material->type == MAT_COLGLASS) {

		// A2 Task 4: Specular refraction
		if (level >= RayDepthMax) {
			return float3(0.0f, 0.0f, 0.0);
		}

		// flip normal and refractive indices if we are exiting object
		// assumes the outside of an object is "air"
		float3 normal = hit.N;
		float3 geoNorm = hit.Ng;
		float etaFrom = 1.0;
		float etaTo = hit.material->eta;
		if (dot(viewDir, normal) < 0) {
			normal = -hit.N;
			geoNorm = -geoNorm;
			etaFrom = etaTo;
			etaTo = 1.0;
		}

		// colour of glass
		float3 brdf = float3(1.0f);

		float3 wi = -viewDir;
		assert(dot(wi, normal) < 0);

		// square root term in refraction equation
		float discriminant = 1 - (powf((etaFrom / etaTo), 2) * (1 - powf(dot(wi, normal), 2)));

		// total internal reflection if no solution
		if (discriminant < 0) {
			assert(etaFrom > etaTo);

			// reflect ray
			float3 wr = -2 * dot(wi, normal) * normal + wi;
			assert(dot(normal, wr) > 0);

			// account for edge case we're reflection goes through plane
			if (dot(geoNorm, wr) < 0) {
				wr = wr - (2 * dot(wr, geoNorm) * geoNorm);
			}
			assert(dot(geoNorm, wr) > 0);
			
			// reflected ray hits something
			Ray reflectedRay(hit.P, wr);
			HitInfo reflectedHit;
			if (globalScene.intersect(reflectedHit, reflectedRay, Epsilon)) {

				// PROJECT: Tint Glass to material colour
				if (hit.material->type == MAT_COLGLASS) {
					brdf = hit.material->BRDF(wr, viewDir, normal);
				}
				return brdf * shade(reflectedHit, -wr, level + 1); 
			}

			// reflected ray hits environment map
			if (globalScene.envMap) {
				return ibl(globalScene.envMap, wr);
			}
			return float3(0.0, 0.0, 0.0f);
		}

		// refracted ray through surface
		float3 wt = (etaFrom / etaTo) * (wi - dot(wi, normal) * normal) - sqrtf(discriminant) * normal;
		assert(dot(normal, wt) < 0);

		// account for edge case where refracted ray doesn't go through surface
		if (dot(geoNorm, wt) > 0) {
			wt = wt - (2 * dot(wt, geoNorm) * geoNorm);
		}
		assert(dot(geoNorm, wt) < 0);

		// Fresnel reflection
		float cosAngleIn = dot(wi, -normal) / (length(wi) * length(-normal));
		float cosAngleOut = dot(wt, -normal) / (length(wt) * length(-normal));
		float ps = (etaFrom * cosAngleIn - etaTo * cosAngleOut) /
			       (etaFrom * cosAngleIn + etaTo * cosAngleOut);
		float pt = (etaFrom * cosAngleOut - etaTo * cosAngleIn) /
			       (etaFrom * cosAngleOut + etaTo * cosAngleIn);
		float FresRatio = 0.5 * (ps * ps + pt * pt);

		// reflected ray out of surface
		float3 wr = -2 * dot(wi, normal) * normal + wi;
		assert(dot(wr, normal) > 0);

		// edge case to make reflected ray consistent with geometric normal
		if (dot(geoNorm, wr) < 0) {
			wr = wr - (2 * dot(wr, geoNorm) * geoNorm);
		}
		assert(dot(geoNorm, wr) > 0);

		// PROJECT: Tint Glass to material colour
		if (hit.material->type == MAT_COLGLASS) {
			brdf = hit.material->BRDF(wt, viewDir, normal);
		}

		float3 L(0.0f);

		// refracted ray hits something
		Ray refracRay(hit.P, wt);
		HitInfo refracHit;
		if (globalScene.intersect(refracHit, refracRay, Epsilon)) {
			L = (1 - FresRatio) * shade(refracHit, -wt, level + 1);
		}
		// refracted ray hits environment map
		else if (globalScene.envMap) {
			L = (1 - FresRatio) * ibl(globalScene.envMap, wt);
		}
		
		// reflected ray hits something
		Ray reflectedRay(hit.P, wr);
		HitInfo reflectedHit;
		if (globalScene.intersect(reflectedHit, reflectedRay, Epsilon)) {
			L += FresRatio * shade(reflectedHit, -wr, level + 1);
		}

		// reflected ray hits environment map
		else if (globalScene.envMap) {
			L += FresRatio * ibl(globalScene.envMap, wr);
		}

		return brdf * L;
	} else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}
}

// Continuous based particle-mesh collision detection
bool moveCollideMesh(float3& offset, float3 position, float3 prevPosition, const TriangleMesh* ignoreMesh = NULL) {
	Ray moveRay(prevPosition, position - prevPosition);
	HitInfo moveHit;
	if (globalScene.intersect(moveHit, moveRay, Epsilon, 1.0f, true, ignoreMesh)) {
		float3 normal = moveHit.Ng;
		if (dot(normal, moveRay.d) > 0) {
			normal *= -1;
		}

		offset = dot(normal, moveHit.P - position) * normal;
		offset += Epsilon * normal;

		return true;
	}
	return false;
}

void releaseParticle() {
	grabbedBody = NULL;
	grabbedParticleIndex = -1;
}

// PROJECT: Dragging
// Get closest particle to mouse and set it to move towards mouse
bool grabParticle() {

	// create ray pointing towards mouse
	HitInfo mouseHit;
	Ray mouseRay = globalScene.eyeRay((int)m_mouseX, globalHeight - (int)m_mouseY);

	// grab if ray hits a softbody
	if (globalScene.intersect(mouseHit, mouseRay, 0.0, FLT_MAX, true)) {
		if (mouseHit.mesh->body) {

			// get closest particle in mesh to mouse
			float minDist = FLT_MAX;
			float tempDist;
			for (int i = 0, i_n = mouseHit.mesh->body->particles.size(); i < i_n; i++) {
				Particle& tempP = mouseHit.mesh->body->particles[i];
				tempDist = length(tempP.position - mouseHit.P);
				if (tempDist < minDist) {

					minDist = tempDist;
					grabbedBody = mouseHit.mesh->body;
					grabbedParticleIndex = i;
					tGrab = mouseHit.t;  // keep particle same distance from camera
				}
			}
			return true;

		}
	}
	return false;
}
// return position particle moves towards while being dragged
float3 grabPosition() {
	Ray ray = globalScene.eyeRay((int)m_mouseX, globalHeight - (int)m_mouseY);
	return ray.o + tGrab * ray.d;
}

// Helper function for computing sqare root of linalg matrix
static void matrixToArray(const float3x3& M, double(&A)[3][3]) {
	// By row then column
	A[0][0] = M.x.x;
	A[0][1] = M.y.x;
	A[0][2] = M.z.x;
	A[1][0] = M.x.y;
	A[1][1] = M.y.y;
	A[1][2] = M.z.y;
	A[2][0] = M.x.z;
	A[2][1] = M.y.z;
	A[2][2] = M.z.z;
}

float3x3 arrayToMatrix(double(&A)[3][3]) {
	float3x3 M;
	M.x.x = A[0][0];
	M.y.x = A[0][1];
	M.z.x = A[0][2];
	M.x.y = A[1][0];
	M.y.y = A[1][1];
	M.z.y = A[1][2];
	M.x.z = A[2][0];
	M.y.z = A[2][1];
	M.z.z = A[2][2];
	return M;
}

// Diagonalization for finding square root of matrix, taken from:
// https://stackoverflow.com/questions/4372224/fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition
//         Slightly modified version of  Stan Melax's code for 3x3 matrix diagonalization (Thanks Stan!)
static void Diagonalize(const double(&A)[3][3], double(&Q)[3][3], double(&D)[3][3])
{
	// A must be a symmetric matrix.
	// returns Q and D such that 
	// Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
	const int maxsteps = 24;  // certainly wont need that many.
	int k0, k1, k2;
	double o[3], m[3];
	double q[4] = { 0.0,0.0,0.0,1.0 };
	double jr[4];
	double sqw, sqx, sqy, sqz;
	double tmp1, tmp2, mq;
	double AQ[3][3];
	double thet, sgn, t, c;
	for (int i = 0; i < maxsteps; ++i)
	{
		// quat to matrix
		sqx = q[0] * q[0];
		sqy = q[1] * q[1];
		sqz = q[2] * q[2];
		sqw = q[3] * q[3];
		Q[0][0] = (sqx - sqy - sqz + sqw);
		Q[1][1] = (-sqx + sqy - sqz + sqw);
		Q[2][2] = (-sqx - sqy + sqz + sqw);
		tmp1 = q[0] * q[1];
		tmp2 = q[2] * q[3];
		Q[1][0] = 2.0 * (tmp1 + tmp2);
		Q[0][1] = 2.0 * (tmp1 - tmp2);
		tmp1 = q[0] * q[2];
		tmp2 = q[1] * q[3];
		Q[2][0] = 2.0 * (tmp1 - tmp2);
		Q[0][2] = 2.0 * (tmp1 + tmp2);
		tmp1 = q[1] * q[2];
		tmp2 = q[0] * q[3];
		Q[2][1] = 2.0 * (tmp1 + tmp2);
		Q[1][2] = 2.0 * (tmp1 - tmp2);

		// AQ = A * Q
		AQ[0][0] = Q[0][0] * A[0][0] + Q[1][0] * A[0][1] + Q[2][0] * A[0][2];
		AQ[0][1] = Q[0][1] * A[0][0] + Q[1][1] * A[0][1] + Q[2][1] * A[0][2];
		AQ[0][2] = Q[0][2] * A[0][0] + Q[1][2] * A[0][1] + Q[2][2] * A[0][2];
		AQ[1][0] = Q[0][0] * A[0][1] + Q[1][0] * A[1][1] + Q[2][0] * A[1][2];
		AQ[1][1] = Q[0][1] * A[0][1] + Q[1][1] * A[1][1] + Q[2][1] * A[1][2];
		AQ[1][2] = Q[0][2] * A[0][1] + Q[1][2] * A[1][1] + Q[2][2] * A[1][2];
		AQ[2][0] = Q[0][0] * A[0][2] + Q[1][0] * A[1][2] + Q[2][0] * A[2][2];
		AQ[2][1] = Q[0][1] * A[0][2] + Q[1][1] * A[1][2] + Q[2][1] * A[2][2];
		AQ[2][2] = Q[0][2] * A[0][2] + Q[1][2] * A[1][2] + Q[2][2] * A[2][2];
		// D = Qt * AQ
		D[0][0] = AQ[0][0] * Q[0][0] + AQ[1][0] * Q[1][0] + AQ[2][0] * Q[2][0];
		D[0][1] = AQ[0][0] * Q[0][1] + AQ[1][0] * Q[1][1] + AQ[2][0] * Q[2][1];
		D[0][2] = AQ[0][0] * Q[0][2] + AQ[1][0] * Q[1][2] + AQ[2][0] * Q[2][2];
		D[1][0] = AQ[0][1] * Q[0][0] + AQ[1][1] * Q[1][0] + AQ[2][1] * Q[2][0];
		D[1][1] = AQ[0][1] * Q[0][1] + AQ[1][1] * Q[1][1] + AQ[2][1] * Q[2][1];
		D[1][2] = AQ[0][1] * Q[0][2] + AQ[1][1] * Q[1][2] + AQ[2][1] * Q[2][2];
		D[2][0] = AQ[0][2] * Q[0][0] + AQ[1][2] * Q[1][0] + AQ[2][2] * Q[2][0];
		D[2][1] = AQ[0][2] * Q[0][1] + AQ[1][2] * Q[1][1] + AQ[2][2] * Q[2][1];
		D[2][2] = AQ[0][2] * Q[0][2] + AQ[1][2] * Q[1][2] + AQ[2][2] * Q[2][2];
		o[0] = D[1][2];
		o[1] = D[0][2];
		o[2] = D[0][1];
		m[0] = fabs(o[0]);
		m[1] = fabs(o[1]);
		m[2] = fabs(o[2]);

		k0 = (m[0] > m[1] && m[0] > m[2]) ? 0 : (m[1] > m[2]) ? 1 : 2; // index of largest element of offdiag
		k1 = (k0 + 1) % 3;
		k2 = (k0 + 2) % 3;
		if (o[k0] == 0.0)
		{
			break;  // diagonal already
		}
		thet = (D[k2][k2] - D[k1][k1]) / (2.0 * o[k0]);
		sgn = (thet > 0.0) ? 1.0 : -1.0;
		thet *= sgn; // make it positive
		t = sgn / (thet + ((thet < 1.E6) ? sqrt(thet * thet + 1.0) : thet)); // sign(T)/(|T|+sqrt(T^2+1))
		c = 1.0 / sqrt(t * t + 1.0); //  c= 1/(t^2+1) , t=s/c 
		if (c == 1.0)
		{
			break;  // no room for improvement - reached machine precision.
		}
		jr[0] = jr[1] = jr[2] = jr[3] = 0.0;
		jr[k0] = sgn * sqrt((1.0 - c) / 2.0);  // using 1/2 angle identity sin(a/2) = sqrt((1-cos(a))/2)  
		jr[k0] *= -1.0; // since our quat-to-matrix convention was for v*M instead of M*v
		jr[3] = sqrt(1.0f - jr[k0] * jr[k0]);
		if (jr[3] == 1.0)
		{
			break; // reached limits of floating point precision
		}
		q[0] = (q[3] * jr[0] + q[0] * jr[3] + q[1] * jr[2] - q[2] * jr[1]);
		q[1] = (q[3] * jr[1] - q[0] * jr[2] + q[1] * jr[3] + q[2] * jr[0]);
		q[2] = (q[3] * jr[2] + q[0] * jr[1] - q[1] * jr[0] + q[2] * jr[3]);
		q[3] = (q[3] * jr[3] - q[0] * jr[0] - q[1] * jr[1] - q[2] * jr[2]);
		mq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
		q[0] /= mq;
		q[1] /= mq;
		q[2] /= mq;
		q[3] /= mq;
	}
}

// returns the square root of a matrix
//     Finds Q and D such that M = Q*D*Q^T
//     Squareroot is given by Q*sqrt(D)*Q^T
// Requires: M is symmetric
float3x3 sqrtM(const float3x3& M) {
	double A[3][3], Q[3][3], D[3][3];
	matrixToArray(M, A);

	// diagonalize
	Diagonalize(A, Q, D);
	float3x3 Qm = arrayToMatrix(Q);
	float3x3 Dm = arrayToMatrix(D);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (Dm[i][j] < 0.0f) {
				Dm[i][j] = 0.0f;
			}
		}
	}
	float3x3 Dmsqrt = sqrt(Dm);
	return mul(Qm, mul(Dmsqrt, transpose(Qm)));
}



// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome to CS488/688!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};

//#define PHYS_TIME
#ifdef PHYS_TIME
long long dur = 0;
#endif

// main window
// you probably do not need to modify this in A0 to A3.
class CS488Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for CS488Window
	OpenGLInit glInit;

	CS488Window() {}
	virtual ~CS488Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
		}
		for (int i = 0; i < globalPhysBodies.bodies.size(); i++) {
			for (int j = 0; j < globalPhysBodies.bodies[i]->spheres.size(); j++) {
				globalScene.debugSpheres.push_back(&(globalPhysBodies.bodies[i]->spheres[j]));
			}
			if (globalPhysBodies.bodies[i]->meshFilePath) {
				globalScene.addObject(&(globalPhysBodies.bodies[i]->mesh));
			}
		}
		globalScene.preCalc();

		if (globalPhysBodies.numBodies > 0) {
			globalPhysBodies.resolveConstraints();
			globalPhysBodies.updateMeshes();
		}

		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles || globalPhysBodies.numBodies > 0) {
				if (globalEnableParticles) {
					globalParticleSystem.step();
				}
				if (globalPhysBodies.numBodies > 0) {

					// MEASURE TIME PER STEP
					#ifdef PHYS_TIME
					auto timeBefore = high_resolution_clock::now();
					#endif 

					globalPhysBodies.step();

					#ifdef PHYS_TIME
					auto timeAfter = high_resolution_clock::now();
					auto ms_int = duration_cast<milliseconds>(timeAfter - timeBefore);
					dur += ms_int.count();
					std::cout << dur / (globalFrameCount+1) << "\n";
					#endif
				}
				
				if (globalRenderType == RENDER_RAYTRACE) {
					globalScene.preCalc();
				}
			}

			// A2 Extra 1: Faster Ray Tracing
			// measure time per frame
			//#define RENDER_TIME
			#ifdef RENDER_TIME
			auto timeBefore = high_resolution_clock::now();
			#endif 

			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}

			#ifdef RENDER_TIME
			auto timeAfter = high_resolution_clock::now();
			auto ms_int = duration_cast<milliseconds>(timeAfter - timeBefore);
			std::cout << ms_int.count() << "\n";
			#endif 

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}
			if (globalVidRecording) {
				char fileName[1024];
				sprintf(fileName, "frames\\vid%03dframe%05d.png", videoId, videoFrameCount);
				videoFrameCount++;
				FrameBuffer.save(fileName);
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};


