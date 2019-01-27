#pragma once
#include "Matrix.h"

// 模型*观察*投影矩阵，用于将顶点/方向矢量从模型空间变换到裁切空间
Matrix4x4 MVP;
// 模型*观察矩阵，用于将顶点/方向矢量从模型空间变换到观察空间
Matrix4x4 MV;
// 观察矩阵，用于将顶点/方向矢量从世界空间变换到观察空间
Matrix4x4 V;
// 透视投影矩阵，用于将顶点/方向矢量从观察空间变换到裁切空间
Matrix4x4 P;
// 观察*投影矩阵， 用于将顶点/方向矢量从世界空间变换到裁切空间
Matrix4x4 VP;
// MV的转置矩阵
Matrix4x4 T_MV;
// MV的逆转置矩阵，用于将法线从模型空间变换到观察空间
Matrix4x4 IT_MV;
// 模型矩阵，用于将顶点/法线从模型空间变换到世界空间
Matrix4x4 Object2World;
// Entity2World的逆矩阵，用于将顶点/法线从世界空间变换到模型空间
Matrix4x4 World2Object;
// 正交投影矩阵，用于将顶点/方向矢量从观察空间变换到裁切空间
Matrix4x4 O;
