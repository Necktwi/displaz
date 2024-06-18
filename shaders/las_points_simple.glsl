#version 150
// Copyright 2015, Christopher J. Foster and the other displaz contributors.
// Use of this code is governed by the BSD-style license found in LICENSE.txt

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelViewProjectionMatrix;

//------------------------------------------------------------------------------
#if defined(VERTEX_SHADER)

uniform float pointRadius = 0.1;    //# uiname=Point Radius; min=0.001; max=10
uniform float trimRadius = 1000000; //# uiname=Trim Radius; min=1; max=1000000
uniform float reference = 400.0;    //# uiname=Reference Intensity; min=0.001; max=100000
uniform float exposure = 1.0;       //# uiname=Exposure; min=0.001; max=10000
uniform float contrast = 1.0;       //# uiname=Contrast; min=0.001; max=10000
uniform int colorMode = 0;          //# uiname=Colour Mode; enum=Intensity|Colour|Return Index|Point Source|Las Classification|File Number|Distance
uniform int selectionMode = 0;      //# uiname=Selection; enum=All|Classified|First Return|Last Return|First Of Several
uniform float minPointSize = 0;
uniform float maxPointSize = 400.0;
// Point size multiplier to get from a width in projected coordinates to the
// number of pixels across as required for gl_PointSize
uniform float pointPixelScale = 0;
uniform vec3 cursorPos = vec3(0);
uniform int fileNumber = 0;
in vec3 position;
in float intensity;
in int returnNumber;
in int numberOfReturns;
in int pointSourceId;
in int classification;
in vec3 color;
in float distance;
//in float heightAboveGround;

void main()
{
    vec4 p = modelViewProjectionMatrix * vec4(position,1.0);
    gl_PointSize = 4;
    gl_Position = p;
}


//------------------------------------------------------------------------------
#elif defined(FRAGMENT_SHADER)


out vec4 fragColor;


void main()
{
    fragColor = vec4(1);
}

#endif

