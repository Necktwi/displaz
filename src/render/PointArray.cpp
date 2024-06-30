// Copyright 2015, Christopher J. Foster and the other displaz contributors.
// Use of this code is governed by the BSD-style license found in LICENSE.txt

#include "PointArray.h"

#include "QtLogger.h"

#include "util.h"
#include "glutil.h"

#include <QGLShaderProgram>
#include <QTime>

#include <functional>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <random>
#include <queue>

#include <cfloat>
#include <chrono>
#include <thread>
#include <iostream>

#include "ply_io.h"

#include "ClipBox.h"


//------------------------------------------------------------------------------
/// Functor to compute octree child node index with respect to some given split
/// point
struct OctreeChildIdx
{
    const V3f* m_P;
    V3f m_center;
    int operator()(size_t i)
    {
        V3f p = m_P[i];
        return 4*(p.z >= m_center.z) +
               2*(p.y >= m_center.y) +
                 (p.x >= m_center.x);
    }

    OctreeChildIdx(const V3f* P, const V3f& center) : m_P(P), m_center(center) {}
};


struct OctreeNode {
   OctreeNode* children[8]; ///< Child nodes - order (x + 2*y + 4*z)
   size_t beginIndex;       ///< Begin index of points in this node
   size_t endIndex;         ///< End index of points in this node
   mutable size_t nextBeginIndex; ///< Next index for incremental rendering
   Imath::Box3f bbox;       ///< Actual bounding box of points in node
   V3f center;              ///< center of the node
   float halfWidth;         ///< Half the axis-aligned width of the node.
   mutable size_t mVboIndex;
   OctreeNode(const V3f& center, float halfWidth)
      : beginIndex(0), endIndex(0), nextBeginIndex(0),
        center(center), halfWidth(halfWidth), mVboIndex(0)
      {
         for (int i = 0; i < 8; ++i)
            children[i] = 0;
      }

   ~OctreeNode()
      {
         for (int i = 0; i < 8; ++i)
            delete children[i];
      }

   size_t findNearest(const EllipticalDist& distFunc,
                      const V3d& offset, const V3f* p,
                      double& dist) const
      {
         return beginIndex + distFunc.findNearest(offset, p + beginIndex,
                                                  endIndex - beginIndex,
                                                  &dist);
      }

   size_t size() const { return endIndex - beginIndex; }

   bool isLeaf() const { return beginIndex != endIndex; }

   /// Estimate cost of drawing a single leaf node with to given camera
   /// position, quality, and incremental settings.
   ///
   /// Returns estimate of primitive draw count and whether there's anything
   /// more to draw.
   DrawCount drawCount(const V3f& relCamera,
                       double quality, bool incrementalDraw) const
      {
         assert(isLeaf());
         const double drawAllDist = 100;
         double dist = (this->bbox.center() - relCamera).length();
         double diagRadius = this->bbox.size().length()/2;
         // Subtract bucket diagonal dist, since we really want an approx
         // distance to closest point in the bucket, rather than dist to center.
         dist = std::max(10.0, dist - diagRadius);
         double desiredFraction = std::min(1.0, quality*pow(drawAllDist/dist, 2));
         size_t chunkSize = (size_t)ceil(this->size()*desiredFraction);
         DrawCount drawCount;
         drawCount.numVertices = chunkSize;
         if (incrementalDraw)
         {
            drawCount.numVertices = (this->nextBeginIndex >= this->endIndex) ? 0 :
               std::min(chunkSize, this->endIndex - this->nextBeginIndex);
         }
         drawCount.moreToDraw = this->nextBeginIndex < this->endIndex;
         return drawCount;
      }
};


struct ProgressFunc
{
    PointArray& points;
    size_t totProcessed;

    ProgressFunc(PointArray& points) : points(points), totProcessed(0) {}

    void operator()(size_t additionalProcessed)
    {
        totProcessed += additionalProcessed;
        emit points.loadProgress(int(100*totProcessed/points.pointCount()));
    }
};

/// Create an octree over the given set of points with position P
///
/// The points for consideration in the current node are the set
/// P[inds[beginIndex..endIndex]]; the tree building process sorts the inds
/// array in place so that points for the output leaf nodes are held in
/// the range P[inds[node.beginIndex, node.endIndex)]].  center is the central
/// split point for splitting children of the current node; radius is the
/// current node radius measured along one of the axes.
static OctreeNode* makeTree(int depth, size_t* inds,
                            size_t beginIndex, size_t endIndex,
                            const V3f* P, const V3f& center,
                            float halfWidth, ProgressFunc& progressFunc)
{
    OctreeNode* node = new OctreeNode(center, halfWidth);
    const size_t pointsPerNode = 1;//1; //100000;
    // Limit max depth of tree to prevent infinite recursion when
    // greater than pointsPerNode points lie at the same position in
    // space.  floats effectively have 24 bit of precision in the
    // mantissa, so there's never any point splitting more than 24 times.
    const int maxDepth = 24;
    size_t* beginPtr = inds + beginIndex;
    size_t* endPtr = inds + endIndex;
    if (endIndex - beginIndex <= pointsPerNode || depth >= maxDepth)
    {
        static std::random_device rd;
        static std::mt19937 g(rd());
        std::shuffle(beginPtr, endPtr, g);

        // Leaf node: set up indices into point list 
        for (size_t i = beginIndex; i < endIndex; ++i)
            node->bbox.extendBy(P[inds[i]]);
        node->beginIndex = beginIndex;
        node->endIndex = endIndex;
        progressFunc(endIndex - beginIndex);
        return node;
    }
    // Partition points into the 8 child nodes
    size_t* childRanges[9] = {0};
    multi_partition(beginPtr, endPtr, OctreeChildIdx(P, center), &childRanges[1], 8);
    childRanges[0] = beginPtr;
    // Recursively generate child nodes
    float h = halfWidth/2;
    for (int i = 0; i < 8; ++i)
    {
        size_t childBeginIndex = childRanges[i]   - inds;
        size_t childEndIndex   = childRanges[i+1] - inds;
        if (childEndIndex == childBeginIndex)
            continue;
        V3f c = center + V3f((i     % 2 == 0) ? -h : h,
                             ((i/2) % 2 == 0) ? -h : h,
                             ((i/4) % 2 == 0) ? -h : h);
        node->children[i] = makeTree(depth+1, inds, childBeginIndex,
                                     childEndIndex, P, c, h, progressFunc);
        node->bbox.extendBy(node->children[i]->bbox);
    }
    return node;
}


//------------------------------------------------------------------------------
// PointArray implementation
PointArray::PointArray()
    : m_npoints(0),
    m_positionFieldIdx(-1),
      m_P(0),
      currentInd(0),
      renderAt(0)
{
   nowtm = time(NULL);
   before = nowtm;
}


PointArray::~PointArray()
{ }


/// Load point cloud in text format, assuming fields XYZ
bool PointArray::loadText(QString fileName, size_t maxPointCount,
                          std::vector<GeomField>& fields, V3d& offset,
                          size_t& npoints, uint64_t& totalPoints)
{
    V3d Psum(0);
    // Use C file IO here, since it's about 40% faster than C++ streams for
    // large text files (tested on linux x86_64, gcc 4.6.3).
    File inFile = fopen(fileName.toUtf8(), "r");
    if (!inFile)
        return false;
    fseek(inFile, 0, SEEK_END);
    const size_t numBytes = ftell(inFile);
    fseek(inFile, 0, SEEK_SET);
    std::vector<Imath::V3d> points;
    Imath::V3d p;
    size_t readCount = 0;
    // Read three doubles; "%*[^\n]" discards up to just before end of line
    while (fscanf(inFile, " %lf %lf %lf%*[^\n]", &p.x, &p.y, &p.z) == 3)
    {
        points.push_back(p);
        ++readCount;
        if (readCount % 10000 == 0)
            emit loadProgress(int(100*ftell(inFile)/numBytes));
    }
    totalPoints = points.size();
    npoints = points.size();
    // Zero points + nonzero bytes => bad text file
    if (totalPoints == 0 && numBytes != 0)
        return false;
    if (totalPoints > 0)
        offset = points[0];
    fields.push_back(GeomField(TypeSpec::vec3float32(), "position", npoints));
    V3f* position = (V3f*)fields[0].as<float>();
    for (size_t i = 0; i < npoints; ++i)
        position[i] = points[i] - offset;
    return true;
}


/// Load ascii version of the point cloud library PCD format
bool PointArray::loadPly(QString fileName, size_t maxPointCount,
                         std::vector<GeomField>& fields, V3d& offset,
                         size_t& npoints, uint64_t& totalPoints)
{
    std::unique_ptr<t_ply_, int(*)(p_ply)> ply(
            ply_open(fileName.toUtf8().constData(), logRplyError, 0, NULL), ply_close);
    if (!ply || !ply_read_header(ply.get()))
        return false;
    // Parse out header data
    p_ply_element vertexElement = findVertexElement(ply.get(), npoints);
    if (vertexElement)
    {
        if (!loadPlyVertexProperties(fileName, ply.get(), vertexElement, fields, offset, npoints))
            return false;
    }
    else
    {
        if (!loadDisplazNativePly(fileName, ply.get(), fields, offset, npoints))
            return false;
    }
    totalPoints = npoints;
    return true;
}


bool PointArray::loadFile(QString fileName, size_t maxPointCount)
{
    QTime loadTimer;
    loadTimer.start();
    setFileName(fileName);
    // Read file into point data fields.  Use very basic file type detection
    // based on extension.
    uint64_t totalPoints = 0;
    V3d offset(0);
    emit loadStepStarted("Reading file");
    if (fileName.toLower().endsWith(".las") || fileName.toLower().endsWith(".laz"))
    {
        if (!loadLas(fileName, maxPointCount, m_fields, offset, m_npoints, totalPoints))
            return false;
    }
    else if (fileName.toLower().endsWith(".ply"))
    {
        if (!loadPly(fileName, maxPointCount, m_fields, offset, m_npoints, totalPoints))
            return false;
    }
#if 0
    else if (fileName.toLower().endsWith(".dat"))
    {
        // Load crappy db format for debugging
        std::ifstream file(fileName.toUtf8(), std::ios::binary);
        file.seekg(0, std::ios::end);
        totalPoints = file.tellg()/(4*sizeof(float));
        file.seekg(0);
        m_fields.push_back(GeomField(TypeSpec::vec3float32(), "position", totalPoints));
        m_fields.push_back(GeomField(TypeSpec::float32(), "intensity", totalPoints));
        float* position = m_fields[0].as<float>();
        float* intensity = m_fields[1].as<float>();
        for (size_t i = 0; i < totalPoints; ++i)
        {
            file.read((char*)position, 3*sizeof(float));
            file.read((char*)intensity, 1*sizeof(float));
            bbox.extendBy(V3d(position[0], position[1], position[2]));
            position += 3;
            intensity += 1;
        }
        m_npoints = totalPoints;
    }
#endif
    else
    {
        // Last resort: try loading as text
        if (!loadText(fileName, maxPointCount, m_fields, offset, m_npoints, totalPoints))
            return false;
    }
    // Search for position field
    m_positionFieldIdx = -1;
    for (size_t i = 0; i < m_fields.size(); ++i)
    {
        if (m_fields[i].name == "position" && m_fields[i].spec == TypeSpec::vec3float32())
        {
            m_positionFieldIdx = (int)i;
            break;
        }
    }
    /*
    m_Tris.push_back(0);
    m_Tris.push_back(1);
    m_Tris.push_back(2);
    for (size_t i = 3; i < 1000; ++i)
    {
       m_Tris.push_back(i-2);
       m_Tris.push_back(i-1);
       m_Tris.push_back(i);
    }
    
       {
          0, 1, 2,
             2, 3, 1,
             1, 5, 7,
             7, 3, 1,
             0, 1,
             }*/
    if (m_positionFieldIdx == -1)
    {
        g_logger.error("No position field found in file %s", fileName);
        return false;
    }
    m_P = (V3f*)m_fields[m_positionFieldIdx].as<float>();
/*
    m_P[ 0]={00.0, 00.0, 00.0};
    m_P[ 1]={30.0, 00.0, 00.0};
    m_P[ 2]={00.0, 30.0, 00.0};
    m_P[ 3]={30.0, 30.0, 00.0};
    m_P[ 4]={00.0, 00.0, 30.0};
    m_P[ 5]={30.0, 00.0, 30.0};
    m_P[ 6]={00.0, 30.0, 30.0};
    m_P[ 7]={30.0, 30.0, 30.0};
    m_P[ 8]={10.0, 10.0, 10.0};
    m_P[ 9]={20.0, 10.0, 10.0};
    m_P[10]={10.0, 20.0, 10.0};
    m_P[11]={20.0, 20.0, 10.0};
    m_P[12]={10.0, 10.0, 20.0};
    m_P[13]={20.0, 10.0, 20.0};
    m_P[14]={10.0, 20.0, 20.0};
    m_P[15]={20.0, 20.0, 20.0};
    m_P[16]={13.3, 13.3, 13.3};
    m_P[17]={16.6, 13.3, 13.3};
    m_P[18]={13.3, 16.6, 13.3};
    m_P[19]={16.6, 16.6, 13.3};
    m_P[20]={13.3, 13.3, 16.6};
    m_P[21]={16.6, 13.3, 16.6};
    m_P[22]={13.3, 16.6, 16.6};
    m_P[23]={16.6, 16.6, 16.6};
*/
    m_P[ 0]={00.0, 00.0, 00.0};
    m_P[ 1]={30.0, 00.0, 00.0};
    m_P[ 2]={00.0, 30.0, 00.0};
    m_P[ 3]={30.0, 30.0, 00.0};
    m_P[ 4]={10.0, 10.0, 00.0};
    m_P[ 5]={20.0, 10.0, 00.0};
    m_P[ 6]={10.0, 20.0, 00.0};
    m_P[ 7]={20.0, 20.0, 00.0};
    m_P[ 8]={13.3, 13.3, 00.0};
    m_P[ 9]={16.6, 13.3, 00.0};
    m_P[10]={13.3, 16.6, 00.0};
    m_P[11]={16.6, 16.6, 00.0};

// Compute bounding box and centroid
    Imath::Box3d bbox;
    V3d centroid(0);
    V3d Psum(0);
    for (size_t i = 0; i < m_npoints; ++i) {
        Psum += m_P[i];
        bbox.extendBy(m_P[i]);
    }
    if (m_npoints > 0)
        centroid = (1.0/m_npoints) * Psum;
    centroid += offset;
    bbox.min += offset;
    bbox.max += offset;

    setBoundingBox(bbox);
    setOffset(offset);
    setCentroid(centroid);
    emit loadProgress(100);
    g_logger.info("Loaded %d of %d points from file %s in %.2f seconds",
                  m_npoints, totalPoints, fileName, loadTimer.elapsed()/1000.0);
    if (totalPoints == 0)
    {
        m_rootNode.reset(new OctreeNode(V3f(0), 1));
        return true;
    }
    // Sort points into octree order
    emit loadStepStarted("Sorting points");
    std::unique_ptr<size_t[]> inds(new size_t[m_npoints]);
    for (size_t i = 0; i < m_npoints; ++i)
        inds[i] = i;
    // Expand the bound so that it's cubic.  Not exactly sure it's required
    // here, but cubic nodes sometimes work better the points are better
    // distributed for LoD, splitting is unbiased, etc.
    Imath::Box3f rootBound(bbox.min - offset, bbox.max - offset);
    V3f diag = rootBound.size();
    float rootRadius = std::max(std::max(diag.x, diag.y), diag.z) / 2;
    ProgressFunc progressFunc(*this);
    m_rootNode.reset(makeTree(0, &inds[0], 0, m_npoints, &m_P[0],
                              rootBound.center(), rootRadius, progressFunc));
    // Reorder point fields into octree order
    emit loadStepStarted("Reordering fields");
    for (size_t i = 0; i < m_fields.size(); ++i)
    {
        g_logger.debug("Reordering field %d: %s", i, m_fields[i]);
        reorder(m_fields[i], inds.get(), m_npoints);
        emit loadProgress(int(100*(i+1)/(m_fields.size()+1))); // denominator +1 for permutation reorder below
    }
    m_P = (V3f*)m_fields[m_positionFieldIdx].as<float>();

    // The index we want to store is the reverse permutation of the index above
    // This is necessary if we want to mutate the data later
    m_inds = std::unique_ptr<uint32_t[]>(new uint32_t[m_npoints]);
    for (size_t i = 0; i < m_npoints; ++i) m_inds[inds[i]] = i; // Works for m_npoints < UINT32_MAX
    emit loadProgress(int(100));

    return true;
}


void PointArray::mutate(std::shared_ptr<GeometryMutator> mutator)
{
    // Now we need to find the matching columns
    auto npoints = mutator->pointCount();
    const std::vector<GeomField>& mutFields = mutator->fields();
    auto mutIdx = mutator->index();

    if (m_npoints > UINT32_MAX)
    {
        g_logger.error("Mutation with more than 2^32 points is not supported");
        return;
    }

    // Check index is valid
    for (size_t j = 0; j < npoints; ++j)
    {
        if (mutIdx[j] >= m_npoints)
        {
            g_logger.error("Index out of bounds - got %d (should be between zero and %d)", mutIdx[j], m_npoints-1);
            return;
        }
    }

    for (size_t mutFieldIdx = 0; mutFieldIdx < mutFields.size(); ++mutFieldIdx)
    {
        if (mutFields[mutFieldIdx].name == "index")
            continue;

        // Attempt to find a matching index
        int foundIdx = -1;
        for (size_t fieldIdx = 0; fieldIdx < m_fields.size(); ++fieldIdx)
        {
            if (m_fields[fieldIdx].name == mutFields[mutFieldIdx].name)
            {
                if (!(m_fields[fieldIdx].spec == mutFields[mutFieldIdx].spec))
                {
                    g_logger.warning("Fields with name \"%s\" do not have matching types, skipping.", m_fields[fieldIdx].name);
                    break;
                }

                if (mutFields[mutFieldIdx].name == "position")
                    g_logger.warning("Moving points by large distances may result in visual artefacts");

                foundIdx = (int)fieldIdx;
                break;
            }
        }

        if (foundIdx == -1)
        {
            g_logger.warning("Couldn't find a field labeled \"%s\"", mutFields[mutFieldIdx].name);
            continue;
        }

        if (mutFields[mutFieldIdx].name == "position")
        {
            assert(m_fields[foundIdx].spec == TypeSpec::float32());
            // Special case for floating point position with offset.
            float* dest = m_fields[foundIdx].as<float>();
            const float* src = mutFields[mutFieldIdx].as<float>();
            V3d off = offset() - mutator->offset();
            for (size_t j = 0; j < npoints; ++j)
            {
                float* d = &dest[3*m_inds[mutIdx[j]]];
                const float* s = &src[3*j];
                d[0] = s[0] - off.x;
                d[1] = s[1] - off.y;
                d[2] = s[2] - off.z;
            }
        }
        else
        {
            // Now we copy data from the mutator to the object
            char* dest = m_fields[foundIdx].data.get();
            char* src = mutFields[mutFieldIdx].data.get();
            size_t fieldsize = m_fields[foundIdx].spec.size();
            for (size_t j = 0; j < npoints; ++j)
            {
                memcpy(dest + fieldsize*m_inds[mutIdx[j]], src + fieldsize*j, fieldsize);
            }
        }
    }
}


bool PointArray::pickVertex(const V3d& cameraPos,
                            const EllipticalDist& distFunc,
                            V3d& pickedVertex,
                            double* distance,
                            std::string* info) const
{
    if (m_npoints == 0)
        return false;

    double closestDist = DBL_MAX;
    size_t closestIdx = 0;

    typedef std::pair<double, const OctreeNode*> PriorityNode;

    auto makePriortyNode = [&](const OctreeNode* node)
    {
        // Create (priority,node) pair with priority given by lower bound of
        // distance for pickVertex() vertex search.
        Box3d bbox(offset() + node->bbox.min, offset() + node->bbox.max);
        return PriorityNode(distFunc.boundNearest(bbox), node);
    };

    // Search for the closest point by putting nodes into a priority queue,
    // with closer nodes having higher priority.  Keep track of the current
    // closest point; as soon as the next priority node is further away than
    // this, we're done.
    std::priority_queue<PriorityNode, std::vector<PriorityNode>,
                        std::greater<PriorityNode>> pendingNodes;
    pendingNodes.push(makePriortyNode(m_rootNode.get()));
    while (!pendingNodes.empty())
    {
        auto nextNode = pendingNodes.top();
        double nextMinDist = nextNode.first;
        if (nextMinDist > closestDist)
            break;
        const OctreeNode* node = nextNode.second;
        pendingNodes.pop();

        if (!node->isLeaf())
        {
            for (int i = 0; i < 8; ++i)
            {
                OctreeNode* n = node->children[i];
                if (n)
                    pendingNodes.push(makePriortyNode(n));
            }
        }
        else
        {
            double dist = 0;
            size_t idx = node->findNearest(distFunc, offset(), m_P, dist);
            if(dist < closestDist)
            {
                closestDist = dist;
                closestIdx = idx;
            }
        }
    }

    if(closestDist == DBL_MAX)
        return false;

    *distance = closestDist;
    pickedVertex = V3d(m_P[closestIdx]) + offset();

    if (info)
    {
        // Format all selected point attributes for user display
        // TODO: Make the type dispatch machinary generic & put in TypeSpec
        std::ostringstream out;
        for (size_t i = 0; i < m_fields.size(); ++i)
        {
            const GeomField& field = m_fields[i];
            tfm::format(out, "  %s = ", field.name);
            if (field.name == "position")
            {
                // Special case for position, since it has an associated offset
                const float* p = (float*)(field.data.get() + closestIdx*field.spec.size());
                tfm::format(out, "%.3f %.3f %.3f\n",
                            p[0] + offset().x, p[1] + offset().y, p[2] + offset().z);
            }
            else
            {
                field.format(out, closestIdx);
                tfm::format(out, "\n");
            }
        }
        *info = out.str();
    }

    return true;
}


void PointArray::estimateCost(const TransformState& transState,
                              bool incrementalDraw, const double* qualities,
                              DrawCount* drawCounts, int numEstimates) const
{
    TransformState relativeTrans = transState.translate(offset());
    V3f relCamera = relativeTrans.cameraPos();
    ClipBox clipBox(relativeTrans);

    std::vector<const OctreeNode*> nodeStack;
    nodeStack.push_back(m_rootNode.get());
    while (!nodeStack.empty())
    {
        const OctreeNode* node = nodeStack.back();
        nodeStack.pop_back();
        if (clipBox.canCull(node->bbox))
            continue;
        if (!node->isLeaf())
        {
            for (int i = 0; i < 8; ++i)
            {
                OctreeNode* n = node->children[i];
                if (n)
                    nodeStack.push_back(n);
            }
            continue;
        }
        for (int i = 0; i < numEstimates; ++i)
        {
            drawCounts[i] += node->drawCount(relCamera, qualities[i],
                                             incrementalDraw);
        }
    }
}


static void drawTree(QGLShaderProgram& prog, const TransformState& transState, const OctreeNode* node)
{
    Imath::Box3f bbox(node->center - Imath::V3f(node->halfWidth),
                      node->center + Imath::V3f(node->halfWidth));

    drawBox(transState, bbox, Imath::C3f(1), prog.programId());
    drawBox(transState, node->bbox, Imath::C3f(1,0,0), prog.programId());

    for (int i = 0; i < 8; ++i)
    {
        OctreeNode* n = node->children[i];
        if (n)
            drawTree(prog, transState, n);
    }
}

void PointArray::drawTree(QGLShaderProgram& prog, const TransformState& transState) const
{
    ::drawTree(prog, transState, m_rootNode.get());
}

void PointArray::initializeGL()
{
    Geometry::initializeGL();

    GLuint vao;
    glGenVertexArrays(1, &vao);
    setVAO("points", vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    setVBO("point_buffer", vbo);

    GLuint ebo;
    glGenBuffers(1, &ebo);
    setEBO("element_buffer", ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Tris.size()*sizeof(unsigned int), NULL, GL_STREAM_DRAW);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Tris.size()*sizeof(unsigned int), &m_Tris[0], GL_STATIC_DRAW);
        
}

void PointArray::draw(const TransformState& transState, double quality) const
{
}

bool ndMakesValidTri (std::vector<const OctreeNode*>& triNds,
                      const OctreeNode* sibNd) {
   return true;
}

double volume(V3f& a, V3f& b, V3f& c, V3f& d) {
    return std::abs(
       (a.x*(b.y*c.z + c.y*d.z + d.y*b.z - b.y*d.z - c.y*b.z - d.y*c.z) +
        a.y*(b.z*c.x + c.z*d.x + d.z*b.x - b.z*d.x - c.z*b.x - d.z*c.x) +
        a.z*(b.x*c.y + c.x*d.y + d.x*b.y - b.x*d.y - c.x*b.y - d.x*c.y)) / 6.0);
}

bool isInsideTetrahedron(V3f& a, V3f& b, V3f& c, V3f& d, V3f& p) {
    double V = volume(a, b, c, d);
    double V1 = volume(p, b, c, d);
    double V2 = volume(a, p, c, d);
    double V3 = volume(a, b, p, d);
    double V4 = volume(a, b, c, p);
    return std::abs(V - (V1 + V2 + V3 + V4)) < 1e-6;
}

std::vector<unsigned short> ldrNdIndStk;

unsigned int findChittiNeighbours (const OctreeNode* nd, unsigned short mask,
                                   std::vector<const OctreeNode*>& triNds) {
   unsigned int count=0;
   if (nd->isLeaf()) {
      triNds.push_back(nd);
      return 1;
   }
   for (unsigned short i=0; i<8; ++i) {
      if ((i&mask) && nd->children[i]) {
         count+=findChittiNeighbours(nd->children[i], mask, triNds);
      }
   }
   return count;
}

void findNeighbours (
   const OctreeNode* node, unsigned short ndInd,
   std::vector<const OctreeNode*>& parntNdStack,
   std::vector<unsigned short>& parntNdIndStack, unsigned short direction,
   std::vector<const OctreeNode*>& triNds
) {
   const OctreeNode* parntNd = parntNdStack.back();
   unsigned short parntNdInd = parntNdIndStack.back();
   for (unsigned short mask=1; mask<8; ++mask) {
      unsigned short sibInd = (~mask) & ndInd;
      if ((direction | mask)!=direction)
         continue;
      if ((mask&ndInd)==mask) {
         const OctreeNode* sibNd = parntNd->children[sibInd];
         if (sibNd) {
            if (
               ndMakesValidTri(triNds, sibNd)
            ) {
               //g_logger.info("sib: %d", sibInd);
               triNds.push_back(sibNd);
            }
         } else {
            findNeighbours(nullptr, sibInd, parntNdStack, parntNdIndStack,
                           ndInd xor sibInd, triNds);
         }
      } else {
         unsigned short lParntNdInd = parntNdInd;
         unsigned short lNdInd=ndInd;
         const OctreeNode* lParntNd = parntNd;
         std::vector<const OctreeNode*> lParntNdStk = parntNdStack;
         std::vector<unsigned short> lParntNdIndStk = parntNdIndStack;
         unsigned int treeHeight = parntNdStack.size();
         OctreeNode* csnNd = nullptr;
         unsigned short csnInd = 0;
         unsigned int csnHt = 0;
         unsigned int shlwHt;
         int i=0;
         bool cmnFnd = false;
         --treeHeight;
         if (!treeHeight)
            goto vertInsEnd;
         lNdInd = parntNdIndStack[treeHeight];
         lParntNd = parntNdStack[treeHeight-1];
         // get parentnode whose child is mask compatible
         while (treeHeight>1 && ((ndInd|lNdInd)&mask) != mask) {
            --treeHeight;
            lNdInd = parntNdIndStack[treeHeight];
            lParntNd = parntNdStack[treeHeight-1];
         }
         if (((ndInd|lNdInd)&mask) != mask)
            goto vertInsEnd;
         csnInd = lNdInd & (~mask);
         csnNd = lParntNd->children[csnInd];
         if (!csnNd) {
            while (treeHeight<parntNdStack.size()) {
               lParntNdStk.pop_back();
               lParntNdIndStk.pop_back();
               ++treeHeight;
            }
            sibInd = mask;
            goto vertIns;
         }
         csnHt = treeHeight;
         lParntNdStk[csnHt]=csnNd;
         lParntNdIndStk[csnHt]=csnInd;
         ++csnHt;
         if (csnHt<parntNdStack.size())
            csnInd=parntNdIndStack[csnHt];
         else
            csnInd=ndInd;
         csnInd = csnInd xor mask;
         while (csnNd->children[csnInd] ||
                (csnHt>=lParntNdStk.size() && !csnNd->isLeaf())) {
            csnNd=csnNd->children[csnInd];
            if (csnNd && csnHt<lParntNdStk.size()) {
               lParntNdStk[csnHt]=csnNd;
               lParntNdIndStk[csnHt]=csnInd;
            } else {
               if (csnNd && !csnNd->isLeaf())
                  csnHt+=findChittiNeighbours(csnNd, csnInd, triNds);
               break;
            }
            ++csnHt;
            if (csnHt<parntNdStack.size())
               csnInd=parntNdIndStack[csnHt];
            else
               csnInd=ndInd;
            csnInd = csnInd xor mask;
         }
         printf("sib: %d; ", sibInd);
         sibInd = ndInd xor csnInd;
        vertIns:
         if (ndInd==2 && parntNdInd==2) {
            printf("break\n");
         }
         parntNdIndStack.push_back(ndInd);
         shlwHt = parntNdIndStack.size()<ldrNdIndStk.size()?
            parntNdIndStack.size():ldrNdIndStk.size();
         for (; i<shlwHt; ++i) {
            if (parntNdIndStack[i]!=ldrNdIndStk[i] && !cmnFnd) {
               cmnFnd=true;
            }
            if (cmnFnd) {
               short cNdInd = parntNdIndStack[i];
               short ldrNdInd = ldrNdIndStk[i];
               if ((~cNdInd)&ldrNdInd) {
                  parntNdIndStack.pop_back();
                  goto vertInsEnd;
               }
            }
         }
         parntNdIndStack.pop_back();
         if (csnNd && csnNd->isLeaf())
            triNds.push_back(csnNd);
         else if (csnHt<=lParntNdStk.size()) {
            findNeighbours(nullptr, csnInd, lParntNdStk, lParntNdIndStk,
                           sibInd, triNds);
         }
        vertInsEnd:
         
      }
   }
}

DrawCount PointArray::drawPoints (
   QGLShaderProgram& prog, const TransformState& transState,
   double quality, bool incrementalDraw
) const {
   GLuint vao = getVAO("points");
   glBindVertexArray(vao);

   GLuint vbo = getVBO("point_buffer");
   glBindBuffer(GL_ARRAY_BUFFER, vbo);

   GLuint ebo = getEBO("element_buffer");

   nowtm = time(NULL);
   renderAt=0;
   if(nowtm-before>=2){
      before=nowtm;
      ++currentInd;
      if(currentInd==m_npoints)currentInd=0;
      renderAt=1;
      g_logger.info("currentInd: %ld",currentInd);
   };
  
   TransformState relativeTrans = transState.translate(offset());
   relativeTrans.setUniforms(prog.programId());
   //printActiveShaderAttributes(prog.programId());
   std::vector<ShaderAttribute> activeAttrs = activeShaderAttributes(prog.programId());
   // Figure out shader locations for each point field
   // TODO: attributeLocation() forces the OpenGL usage here to be
   // synchronous.  Does this matter?  (Alternative: bind them ourselves.)
   std::vector<const ShaderAttribute*> attributes;
   for (size_t i = 0; i < m_fields.size(); ++i) {
      const GeomField& field = m_fields[i];
      if (field.spec.isArray()) {
         for (int j = 0; j < field.spec.count; ++j) {
            std::string name = tfm::format("%s[%d]", field.name, j);
            attributes.push_back(findAttr(name, activeAttrs));
         }
      } else {
         attributes.push_back(findAttr(field.name, activeAttrs));
      }
   }
   // Zero out active attributes in case they don't have associated fields
   GLfloat zeros[16] = {0};
   for (size_t i = 0; i < activeAttrs.size(); ++i) {
      prog.setAttributeValue((int)i, zeros, activeAttrs[i].rows,
                             activeAttrs[i].cols);
   }
   // Enable attributes which have associated fields
   for (size_t i = 0; i < attributes.size(); ++i) {
      if (attributes[i])
         glEnableVertexAttribArray(attributes[i]->location);
   }

   // Compute number of bytes required to store all attributes of a vertex, in
   // bytes.
   size_t perVertexBytes = 0;
   for (size_t i = 0; i < m_fields.size(); ++i) {
      const GeomField &field = m_fields[i];
      unsigned int arraySize = field.spec.arraySize();
      unsigned int vecSize = field.spec.vectorSize();
      perVertexBytes += arraySize * vecSize * field.spec.elsize; //sizeof(glBaseType(field.spec));
   }

   DrawCount drawCount;
   ClipBox clipBox(relativeTrans);

   // Create a new uninitialized buffer for the current node, reserving
   // enough space for the entire set of vertex attributes which will be
   // passed to the shader.
   //
   // (This new memory area will be bound to the "point_buffer" VBO until
   // the memory is orphaned by calling glBufferData() next time through
   // the loop.  The orphaned memory should be cleaned up by the driver,
   // and this may actually be quite efficient, see
   // http://stackoverflow.com/questions/25111565/how-to-deallocate-glbufferdata-memory
   // http://hacksoflife.blogspot.com.au/2015/06/glmapbuffer-no-longer-cool.html )
   GLsizeiptr nodeBufferSize = perVertexBytes * m_npoints;
   glBufferData(GL_ARRAY_BUFFER, nodeBufferSize, NULL, GL_STREAM_DRAW);

   // Draw points in each bucket, with total number drawn depending on how far
   // away the bucket is.  Since the points are shuffled, this corresponds to
   // a stochastic simplification of the full point cloud.
   V3f relCamera = relativeTrans.cameraPos();
   std::vector<const OctreeNode*> nodeStack;
   std::vector<unsigned short> ndIndStack;
   std::vector<const OctreeNode*> parntNdStack;
   std::vector<unsigned short> parntNdIndStack;
   std::vector<unsigned short> parntNdCntStack;
   nodeStack.push_back(m_rootNode.get());
   ndIndStack.push_back(0);
   GLintptr bufferOffset = 0;
   unsigned int verticesToDraw = m_npoints;
   size_t ndStkInd = 0;
   unsigned int avgNdDist = 10;
   m_Tris.erase(m_Tris.begin(), m_Tris.end());
   while (!nodeStack.empty()) {
      const OctreeNode* node = nodeStack.back();
      const unsigned short ndInd = ndIndStack.back();
      short count;
      nodeStack.pop_back();
      ndIndStack.pop_back();
      if (clipBox.canCull(node->bbox))
         continue;
      if (!node->isLeaf()) {
         if (parntNdCntStack.size()) {
            count = parntNdCntStack.back();
            while (count==0) {
               parntNdStack.pop_back();
               parntNdIndStack.pop_back();
               parntNdCntStack.pop_back();
               count = parntNdCntStack.back();
               --count;
               if (count) {
                  parntNdCntStack.pop_back();
                  parntNdCntStack.push_back(count);
               }
            }
         }
         parntNdStack.push_back(node);
         parntNdIndStack.push_back(ndInd);
         count=0;
         for (int i = 7; i >=0; --i) {
            const OctreeNode* n = node->children[i];
            if (n) {
               nodeStack.push_back(n);
               ndIndStack.push_back(i);
               ++count;
            }
         }
         parntNdCntStack.push_back(count);
         continue;
      }
      count = parntNdCntStack.back();
      while (count==0) {
         parntNdStack.pop_back();
         parntNdIndStack.pop_back();
         parntNdCntStack.pop_back();
         count = parntNdCntStack.back();
         --count;
      }
      parntNdCntStack.pop_back();
      --count;
      parntNdCntStack.push_back(count);
      if (!incrementalDraw)
         node->nextBeginIndex = node->beginIndex;

      // if (ndStkInd>currentInd) {
      //    break;
      // }
      DrawCount nodeDrawCount
         = node->drawCount(relCamera, quality, incrementalDraw);
      drawCount += nodeDrawCount;

      if (nodeDrawCount.numVertices == 0)
         continue;

      if (m_fields.size() < 1)
         continue;
      /*
        g_logger.info("beginIndex: %d",node->beginIndex);
        g_logger.info("endIndex: %d",node->endIndex);
        g_logger.info("nextBeginIndex: %d",node->nextBeginIndex);
        g_logger.info("center: %f,%f,%f",
        node->center.x, node->center.y, node->center.z);
      */
      GLsizeiptr nodeBufferSize = perVertexBytes * nodeDrawCount.numVertices;

      GLintptr fieldOffset = 0;
      for (
         size_t i = 0, k = 0; i < m_fields.size(); k+=m_fields[i].spec.arraySize(), ++i
      ) {

         const GeomField& field = m_fields[i];
         int arraySize = field.spec.arraySize();
         int vecSize = field.spec.vectorSize();

         // TODO?: Could use a single data-array that isn't split into
         // vertex / normal / color / etc. sections, but has interleaved
         // data ?  OpenGL has a stride value in glVertexAttribPointer for
         // exactly this purpose, which should be used for better efficiency
         // here we write only the current attribute data into this the
         // buffer (e.g. all positions, then all colors)
         GLsizeiptr fieldBufferSize =
            arraySize * vecSize * field.spec.elsize * nodeDrawCount.numVertices;

         // Upload raw data for `field` to the appropriate part of the buffer.
         char* bufferData =
            field.data.get() + node->nextBeginIndex*field.spec.size();
         glBufferSubData(GL_ARRAY_BUFFER, bufferOffset, fieldBufferSize,
                         bufferData);
         ///*
         if (i==0 && ndStkInd==currentInd) {
            std::string s=std::to_string(ndInd);
            for (int i=parntNdIndStack.size()-1; i>=0; --i) {
               s += ",";
               s += std::to_string(parntNdIndStack[i]);
            }
            g_logger.info("%d: %d: %s: %f,%f,%f", ndStkInd,
                          node->beginIndex, s.c_str(), ((V3f*)bufferData)->x,
                          ((V3f*)bufferData)->y, ((V3f*)bufferData)->z);
         }
         //*/

         // Tell OpenGL how to interpret the buffer of raw data which was
         // just uploaded.  This should be a single call, but OpenGL spec
         // insanity says we need `arraySize` calls (though arraySize=1
         // for most usage.)
         for (int j = 0; j < arraySize; ++j) {
            if(j>1)
               g_logger.info("ind: %ld, array: %d!",
                             node->beginIndex, j);
            const ShaderAttribute* attr = attributes[k+j];
            if (!attr)
               continue;

            GLintptr arrayElementOffset = fieldOffset + j*field.spec.elsize;

            if (
               attr->baseType == TypeSpec::Int ||
               attr->baseType == TypeSpec::Uint
            ) {
               glVertexAttribIPointer(
                  attr->location, vecSize, glBaseType(field.spec),
                  0, (const GLvoid *)arrayElementOffset);
            } else {
               glVertexAttribPointer(
                  attr->location, vecSize, glBaseType(field.spec),
                  field.spec.fixedPoint, perVertexBytes,
                  (const GLvoid *)arrayElementOffset);
            }
         }

         fieldOffset += fieldBufferSize;
         bufferOffset += fieldBufferSize;
      }
        
    
      node->nextBeginIndex += nodeDrawCount.numVertices;

      glDrawArrays(GL_POINTS, 0, (GLsizei)drawCount.numVertices);

      node->mVboIndex=ndStkInd;
      std::vector<const OctreeNode*> triNds;
      triNds.push_back(node);
      unsigned short parntNdInd =  parntNdIndStack.back();
      findNeighbours(node, ndInd, parntNdStack, parntNdIndStack, 0b111,
                     triNds);
      ldrNdIndStk=parntNdIndStack;
      ldrNdIndStk.push_back(ndInd);
      if (triNds.size()>=3) {
         int ndCnt = triNds.size();
         for (int i=0; i<ndCnt; ++i) {
            m_Tris.push_back(triNds[i]->mVboIndex);
            m_Tris.push_back(triNds[i+1>=ndCnt?i+1-ndCnt:i+1]->mVboIndex);
            m_Tris.push_back(triNds[i+2>=ndCnt?i+2-ndCnt:i+2]->mVboIndex);
         }
      }
      
      ++ndStkInd;
      if(drawCount.numVertices>=verticesToDraw)
         break;
   }
   
   if (m_Tris.size()) { //renderAt) {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
      // glEnable(GL_DEPTH_TEST);
      //  glDepthFunc(GL_LEQUAL);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                   m_Tris.size()*sizeof(unsigned), &m_Tris[0], GL_STATIC_DRAW);
      //glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
      //                m_Tris.size()*sizeof(unsigned int), &m_Tris[0]);
      //glDrawArrays(GL_TRIANGLES, 0, (GLsizei)drawCount.numVertices);
      glDrawElements(
         GL_TRIANGLES,      // mode
         m_Tris.size(),    // count
         GL_UNSIGNED_INT,   // type
         (void*)0           // element array buffer offset
      );
      for (int i=0;i<m_Tris.size();++i)
         g_logger.info("%lu",
                       m_Tris[i]);
      g_logger.info("%d m_Tris drawn",
                    m_Tris.size());
      /*
        for(int i=0; i< m_Tris.size(); ++i) {
        ++m_Tris[i];
        if(m_Tris[i]==m_npoints)
        m_Tris[i]=0;
        }
      */
   }
   //tfm::printf("Drew %d of total points %d, quality %f\n", totDraw, m_npoints, quality);

   // Disable all attribute arrays - leaving these enabled seems to screw with
   // the OpenGL fixed function pipeline in unusual ways.
   for (size_t i = 0; i < attributes.size(); ++i) {
      if (attributes[i])
         glDisableVertexAttribArray(attributes[i]->location);
   }

   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindVertexArray(0);

   return drawCount;
}
