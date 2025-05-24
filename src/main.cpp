#include "cs488.h"
#include <string.h>
CS488Window CS488;

 
// draw something in each frame
static void draw() {
    for (int j = 0; j < globalHeight; j++) {
        for (int i = 0; i < globalWidth; i++) {
            //FrameBuffer.pixel(i, j) = float3(PCG32::rand()); // noise
            //FrameBuffer.pixel(i, j) = float3(0.5f * (cos((i + globalFrameCount) * 0.1f) + 1.0f)); // moving cosine
            // rainbow moving cosine
            FrameBuffer.pixel(i, j) = float3(0.5f * (cos((i+globalFrameCount) * 0.01f) + 1.0f),
                                             0.5f * (cos((i+globalFrameCount) * 0.01f + 2.094f) + 1.0f),
                                             0.5f * (cos((i+globalFrameCount) * 0.01f + 4.189f) + 1.0f));
        }
    }
}
static void A0(int argc, const char* argv[]) {
    // set the function to be called in the main loop
    CS488.process = draw;
}



// setting up lighting
static PointLightSource light;
static void setupLightSource() {
    //light.position = float3(0.5f, 4.0f, 1.0f); // use this for sponza.obj
    light.position = float3(3.0f, 3.0f, 3.0f);
    light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
    globalScene.addLight(&light);
}



// ======== you probably don't need to modify below in A1 to A3 ========
// loading .obj file from the command line arguments
static TriangleMesh mesh;
static Image envMap;
static Image envMapBlur;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        bool objLoadSucceed = mesh.load(argv[1]);
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
        if (argc > 2) {
            bool mapLoadSucceed = envMap.load(argv[2]);
            if (!mapLoadSucceed) {
                printf("Invalid image file.\n");
                printf("Using no environment map\n");
            }
            else {
                globalScene.envMap = &envMap;

                // A2 Extra 2: Image based lighting
                envMapBlur = envMap.lowPassFilter(4);
                globalScene.envMapBlur = &envMapBlur;
            }
        }
    } else {
        printf("Specify .obj and image file in the command line arguments. Example: CS488.exe cornellbox.obj uffizi_probe.hdr\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    globalScene.addObject(&mesh);
}
static void A1(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RASTERIZE;
}

static void A2(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RAYTRACE;
}

static void A3(int argc, const char* argv[]) {
    globalEnableParticles = true;
    globalEnableGravity = true;  // Task 1
    particleBox = false;         // Task 2
    particleConstrain = true;   // Task 3
    particleGravField = false;   // Task 4
    shading = true;
    shadows = false;

    setupLightSource();
    globalRenderType = RENDER_RAYTRACE; // RENDER_RASTERIZE;
    if (argc > 1) {
        setupScene(argc, argv);
    }
    globalParticleSystem.sphereMeshFilePath = "..\\media\\smallsphere.obj";
    globalParticleSystem.initialize();
}
// ======== you probably don't need to modify above in A1 to A3 ========

// ========= Project functions =======================
 
// setting up lighting
static void setupMyLights() {
    static PointLightSource light2;

    light.position = float3(3.0f, 3.0f, 3.0f);
    light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
    globalScene.addLight(&light);

    light2.position = float3(-3.0f, 3.0f, -2.0f);
    light2.wattage = float3(400.0f, 400.0f, 450.0f);
    globalScene.addLight(&light2);
}


// setup tetris scene, full scene has max of 8 objects
// but less can be specified
static void setupTetris(bool opaque = false, int num = 8) {
    num = std::max(0, std::min(8, num));

    static SoftBody T, T2, Z, S, O, L, J, I;
    const float BLOCK = 0.15;
    static std::vector<SoftBody> pieces(8);
    std::vector<char*> paths(8);

    if (!opaque){
        paths[0] = "..\\media\\tetrisL-glass.obj";
        paths[1] = "..\\media\\tetrisI-glass.obj";
        paths[2] = "..\\media\\tetrisO-glass.obj";
        paths[3] = "..\\media\\tetrisT-glass.obj";
        paths[4] = "..\\media\\tetrisZ-glass.obj";
        paths[5] = "..\\media\\tetrisS-glass.obj";
        paths[6] = "..\\media\\tetrisT-glass.obj";
        paths[7] = "..\\media\\tetrisJ-glass.obj";
    }
    else {
        paths[0] = "..\\media\\tetrisL.obj";
        paths[1] = "..\\media\\tetrisI.obj";
        paths[2] = "..\\media\\tetrisO.obj";
        paths[3] = "..\\media\\tetrisT.obj";
        paths[4] = "..\\media\\tetrisZ.obj";
        paths[5] = "..\\media\\tetrisS.obj";
        paths[6] = "..\\media\\tetrisT.obj";
        paths[7] = "..\\media\\tetrisJ.obj";
    }
    

    pieces[0].startRotation = float3(PI / 2, 0, -PI / 2);
    pieces[1].startRotation = float3(PI / 2, 0, 0);
    pieces[2].startRotation = float3(PI / 2, 0, 0);
    pieces[3].startRotation  = float3(PI / 2, 0, 0);
    pieces[4].startRotation = float3(PI / 2, 0, 0);
    pieces[5].startRotation = float3(PI / 2, 0, 0);
    pieces[6].startRotation = float3(PI / 2, 0, PI);
    pieces[7].startRotation  = float3(PI / 2, 0, PI/2);
    

    pieces[0].startPosition = float3(-BLOCK - 0.02, 0.1, 0.0);
    pieces[1].startPosition = float3(BLOCK * 1, BLOCK * 5, 0.0);
    pieces[2].startPosition = float3(-BLOCK - 0.02, BLOCK * 5 + 0.02, 0.0);
    pieces[3].startPosition = float3(0, BLOCK * 7 + 0.02, 0.0);
    pieces[4].startPosition = float3(-BLOCK * 1, BLOCK * 10, 0.0);
    pieces[5].startPosition = float3(BLOCK, BLOCK * 13 + 0.02, 0.0);
    pieces[6].startPosition = float3(-BLOCK * 2, BLOCK * 15, 0.0);
    pieces[7].startPosition = float3(0, BLOCK * 19 + 0.02, 0.0);

    for (int i = 0; i < num; i++) {
        pieces[i].meshFilePath = paths[i];
        pieces[i].initialize();
        globalPhysBodies.addBody(&pieces[i]);
    }

    // container base
    mesh.load("..//media//base-glass.obj");
    //mesh.rotate(float3(PI/2, 0, 0));
    mesh.translate(float3(-BLOCK/2, -0.275, 0));
    globalScene.addObject(&mesh);

    globalEye = float3(0.910914, - 0.657811, 1.467588);
    globalLookat = float3(0.412801, - 0.154540, 0.761469);

    // set container walls
    globalColBox.setSize(float3(-BLOCK/2, 4.8f, 0.0f), float3(4*BLOCK+0.02, 11.0f, BLOCK+0.01));
}

// Creates a falling stack of cubes
// can have 1 to 5 cubes
static void setupStack(bool opaque = false, int num = 5) {
    num = std::max(1, std::min(5, num));

    static std::vector<SoftBody> bodies(num);

    const float maxScale = 1.75f;
    const float minScale = 0.75f;

    for (int i = 0; i < bodies.size(); i++) {
        SoftBody& body = bodies[i];

        if (opaque) {
            body.meshFilePath = "..\\media\\roundCube.obj";
        } else {
            body.meshFilePath = "..\\media\\roundCube-glass.obj";
        }

        float s;
        if (num == 1) s = maxScale;
        else s = (num - 1 - i) * (maxScale - minScale) / (num - 1) + minScale;

        body.startScale = float3(s);
        body.startVelocity = float3(0.0, -0.5, 0.0);
        body.startPosition = float3(0, 0.35*i, 0.0);\
        body.initialize();
        globalPhysBodies.addBody(&body);
    }

    globalColBox.setSize(float3(0.0f, 0.8f, 0.0f), float3(1.5f, 3.0f, 1.5f));
}

static void setupDrag(int num = 3) {
    num = std::max(1, std::min(10, num));

    static std::vector<SoftBody> bodies(num);


    for (int i = 0; i < bodies.size(); i++) {
        auto& body = bodies[i];

        body.meshFilePath = "..\\media\\roundCube.obj";
        body.stiffness = 0.9;
        body.quadWeight = 0.6f;
        body.startVelocity = 2.0f * float3((PCG32::rand() - 0.5f), (PCG32::rand() - 0.5f), (PCG32::rand() - 0.5f));

        // avoid overlap
        float3 p;
        while (true) {
            bool overlap = false;
            p = (float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f)) / 2.0f;

            for (int j = 0; j < i; j++) {
                if (length2(p - bodies[j].startPosition) < 0.09) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) break;
        }

        body.startPosition = p;
        body.initialize();
        globalPhysBodies.addBody(&body);
    }

    globalColBox.setSize(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
}

static void setupSpring() {
    static SoftBody body;

    globalMoveToMouse = true;
    showVerticies = true;
    globalScene.envMap = NULL;

    body.initialize();
    body.stiffness = 0.001f;
    globalPhysBodies.addBody(&body);

    globalColBox.setSize(float3(0.0f, 0.0f, 0.0f), float3(0.75f, 0.75f, 0.75f));
}

static void setupCollide() {
    static SoftBody body;
    static SoftBody body2;

    showVerticies = true;
    globalScene.envMap = NULL;

    body.startPosition = float3(0, 0.61, 0.0);
    body.initialize();
    body.quadWeight = 0.0f;
    body.stiffness = 0.0f;
    body.springDamp = 0.0f;
    globalPhysBodies.addBody(&body);

    body2.meshFilePath = "..\\media\\cornellbox.obj";
    body2.initialize();
    body2.quadWeight = 0.0f;
    body2.stiffness = 1.0f;
    globalPhysBodies.addBody(&body2);

    globalColBox.setSize(float3(0.0f, 0.2f, 0.0f), float3(1.2f, 1.2f, 1.2f));
}

static void setupParticles() {
    static SoftBody body;
    
    showVerticies = true;
    globalScene.envMap = NULL;

    body.initialize();
    body.startVelocity = float3(0.0, -2.0f, 0.0f);
    body.quadWeight = 0.5f;
    body.stiffness = 0.01f;
    globalPhysBodies.addBody(&body);

    globalColBox.setSize(float3(0.0f, 0.0f, 0.0f), float3(0.75f, 0.75f, 0.75f));
}

static void setupDragV() {
    static SoftBody body;

    showVerticies = true;
    globalScene.envMap = NULL;

    body.meshFilePath = "..\\media\\roundCube.obj";
    body.initialize();
    body.quadWeight = 0.8f;
    body.stiffness = 0.1f;
    globalPhysBodies.addBody(&body);

    globalColBox.setSize(float3(0.0f, 0.0f, 0.0f), float3(0.75f, 0.75f, 0.75f));
}

static void setupTest() {
    static SoftBody body;
    static SoftBody body2;

    showVerticies = true;

    mesh.load("..//media//cube.obj");
    globalScene.addObject(&mesh);

    //body.meshFilePath = "..\\media\\roundCube.obj";
    /*body.stiffness = 0.8;
    body.quadWeight = 0.9f;*/
    body.startRotation = float3(PI / 2, 0, 0);
    body.startVelocity = float3(0, 0, 0);
    body.startPosition = float3(0, 0.71, 0.0);
    body.initialize();
    globalPhysBodies.addBody(&body);

    /*body2.meshFilePath = "..\\media\\tetrisI.obj";
    body2.startRotation = float3(PI / 2, 0, 0);
    body2.startVelocity = float3(0.00, 0, 0);
    body2.startPosition = float3(0.15, 0.71, 0.0);
    body2.initialize();
    globalPhysBodies.addBody(&body2);*/

    globalColBox.setSize(float3(0.0725f, 0.8f, 0.0f), float3(2.4f, 2.4f, 2.4f));
}

static void Project(int argc, const char* argv[]) {

    // standard feature toggles
    globalEnableParticles = false;
    globalEnableGravity = true;
    particleBox = true;
    showVerticies = false;
    shading = true;
    shadows = true;

    // aditional setup
    setupMyLights();
    globalRenderType = RENDER_RAYTRACE;
    envMap.load("..\\media\\pisa.hdr");
    globalEnvmType = ENVM_USC;
    globalScene.envMap = &envMap;

    //setupTetris();
    //setupStack();

    if (argc > 1) {
        std::string option = argv[1];
        std::string arg = "";
        if (argc > 2) arg = argv[2];

        if (option == "-tetris") {
            if (arg != "") setupTetris(false, std::stoi(arg));
            else setupTetris(false);
        }
        else if (option == "-tetrisO") {
            if (arg != "") setupTetris(true, std::stoi(arg));
            else setupTetris(true);
        }
        else if (option == "-stack") {
            if (arg != "") setupStack(false, std::stoi(arg));
            else setupStack(false);
        }
        else if (option == "-stackO") {
            if (arg != "") setupStack(true, std::stoi(arg));
            else setupStack(true);
        }
        else if (option == "-drag") {
            if (arg != "") setupDrag(std::stoi(arg));
            else setupDrag();
        }
        else if (option == "-dragV") {
            setupDragV();
        }
        else if (option == "-collide") {
            setupCollide();
        }
        else if (option == "-particles") {
            setupParticles();
        }
        else if (option == "-spring") {
            setupSpring();
        }
        else {
            std::cout << "invalid arguments" << std::endl;
            exit(1);
        }
    }
    else {
        setupTetris();
    }
}


int main(int argc, const char* argv[]) {
    //A0(argc, argv);
    //A1(argc, argv);
    //A2(argc, argv);
    //A3(argc, argv);
    Project(argc, argv);  

    CS488.start();
}

// Debug print statements
static void printV(const float3& v) {
    printf("%f %f %f\n", v.x, v.y, v.z);
}
static void printM(const float3x3& R) {
    printf("%f %f %f\n", R.x.x, R.y.x, R.z.x);
    printf("%f %f %f\n", R.x.y, R.y.y, R.z.y);
    printf("%f %f %f\n\n", R.x.z, R.y.z, R.z.z);
}