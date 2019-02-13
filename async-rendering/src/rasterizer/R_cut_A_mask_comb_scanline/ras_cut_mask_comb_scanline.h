
#pragma once

#include "../shared/ras_base.h"

#include <map>

#include <cuda.h>
#include <host_defines.h>

#include "mochimazui/3rd/gl_4_5_compatibility.h"
#include <cuda_gl_interop.h>
#include "mochimazui/glpp.h"
#include "mochimazui/cuda_array.h"

#include <windows.h>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include "mochimazui/d3dx12.h"
#include "mochimazui/pix3Runtime/pix3.h"

#include <wrl.h>
#include <vector>
#include <shellapi.h>

using namespace DirectX;

using Microsoft::WRL::ComPtr;

/*
// DX12
#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN // 从 Windows 头中排除极少使用的资料
#include <windows.h>
#include <WinUser.h>
#include <tchar.h>
//添加WTL支持 方便使用COM
#include <wrl.h>
using namespace Microsoft;
using namespace Microsoft::WRL;
#include <dxgi1_6.h>
#include <DirectXMath.h>
using namespace DirectX;
//for d3d12
#include <d3d12.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
//linker
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")

#if defined(_DEBUG)
#include <dxgidebug.h>
#endif

#include "..\WindowsCommons\d3dx12.h"

#define GRS_WND_CLASS_NAME _T("Game Windows Clss")
#define GRS_WND_TITLE   _T()("DirectX12 Trigger Sample")

#define GRS_THROW_IF_FAILED(hr) if (FAILED(hr)){ throw CGRSCOMException(hr); }

class CGRSCOMException
{ 
public:	
	CGRSCOMException(HRESULT hr) : m_hrError(hr)	
	{	
	}
	HRESULT Error() const	
	{ 
		return m_hrError; 
	}
private:	const HRESULT m_hrError; 
};


struct GRS_VERTEX

{
	XMFLOAT3 position;
	XMFLOAT4 color;
};
*/

namespace Mochimazui {

class VGContainer;

namespace Rasterizer_R_Cut_A_Mask_Comb_Scanline {

using GLPP::NamedBuffer;
using GLPP::NamedFramebuffer;
using GLPP::NamedTexture;
using GLPP::ShaderProgram;

using CUDATL::CUDAArray;

class VGRasterizer : public RasterizerBase::VGRasterizer {

	typedef RasterizerBase::VGRasterizer _Base;

public:
	VGRasterizer();
	~VGRasterizer();

	void init();
	void uninit();

	void addVg(const VGContainer &vgc);
	void clear() {}

	void setFragmentSize(int s) { _fragSize = s; }

	void rasterizeImpl();

private:
	void initProgram();
	void initBuffer();
	void initFramebuffer();

	void initCommandList();
	void uninitCommandList();

	void onResize(int _width, int _height);

private:
	void initQMMaskTable();

	template <int FRAG_SIZE>
	void rasterizeImpl();

	static const UINT FrameCount = 2;
	static const UINT ThreadCount = 1;
	static const UINT ParticleCount = 10000;        // The number of particles in the n-body simulation.


protected:

	uint32_t _fragSize = 2;

	// for debug.
	bool _dbgDumpWindingNumber = false;
	bool _dbgDumpFragmentData = false;

	struct _GL{
		_GL() {}
		struct _GL_Buffer{
			_GL_Buffer() {}

			NamedBuffer stencilDrawData;
			NamedBuffer stencilDrawMask;

			NamedBuffer outputIndex;
			NamedBuffer outputFragmentData;
			NamedBuffer outputSpanData;
			NamedBuffer outputFillInfo;

			NamedBuffer qm_output_stencil_mask;

			// -- debug --
			NamedBuffer dbgCurveVertex;
			NamedBuffer dbgCurveColor;

			NamedBuffer dbgDrawStencilDump_0;
			NamedBuffer dbgDrawStencilDump_1;
			NamedBuffer dbgDrawStencilDump_2;
		} buffer;

		struct _GL_Texture{
			_GL_Texture() {}

			// texbuffer
			NamedTexture stencilDrawData;
			NamedTexture stencilDrawMask;

			NamedTexture outputIndex;
			NamedTexture outputFragmentData;
			NamedTexture outputSpanData;
			NamedTexture outputFillInfo;

			// tex2D
			NamedTexture stencilDraw;

			// -- debug --
			NamedTexture dbgCurveVertex;
			NamedTexture dbgCurveColor;

			NamedTexture dbgDrawCount;

			NamedTexture dbgDrawStencilDump_0;
			NamedTexture dbgDrawStencilDump_1;
			NamedTexture dbgDrawStencilDump_2;
		} texture;

		struct _GL_Framebuffer{
			_GL_Framebuffer() {}
			NamedFramebuffer stencilDrawMS;
		} framebuffer;

		struct _GL_Program{
			_GL_Program() {}

			ShaderProgram output;

			// -- debug --
			ShaderProgram dbgCurve;
			ShaderProgram dbgCurveFragment;
			ShaderProgram dbgOutputScale;

		} program;

	} _gl;

	struct _GPU_Array{
		_GPU_Array() {}

		// transform && stroke to fill
		CUDAArray<float2> strokeTransformedVertex;
		CUDAArray<int> strokeToFillNewCurveTemp;
		
		CUDAArray<float2> transformedVertex;
		
		// monotonize
		CUDAArray<int> curve_pixel_count;
		CUDAArray<float> monotonic_cutpoint_cache;		
		CUDAArray<float> intersection;

		CUDAArray<float> monoCurveT;
		CUDAArray<uint32_t> monoCurveNumber;
		CUDAArray<uint32_t> monoCurveSize;
		CUDAArray<uint32_t> curveFragmentNumber;

		CUDAArray<int32_t> ic4Context;

		CUDAArray<int32_t> fragmentData;

		// mask
		CUDAArray<uint32_t> amaskTable;
		CUDAArray<uint32_t> pmaskTable;

		// temp for CUDA SM gen stencil
		CUDAArray<int32_t> blockBoundaryBins;

		// for CUDA cell list output
		CUDAArray<int32_t> cellListPos;
		CUDAArray<int32_t> cellListFillInfo;
		CUDAArray<int32_t> cellListMaskIndex;

	} _gpu;

	struct __CUDA {
		__CUDA() {}
		struct __CUDAResrouce {
			__CUDAResrouce() :
				stencilDrawData(nullptr), stencilDrawMask(nullptr),
				outputIndex(nullptr), outputFragment(nullptr),
				outputSpan(nullptr), outputFillInfo(nullptr)
			{}

			cudaGraphicsResource *stencilDrawData = nullptr;
			cudaGraphicsResource *stencilDrawMask = nullptr;

			cudaGraphicsResource *outputIndex = nullptr;
			cudaGraphicsResource *outputFragment = nullptr;
			cudaGraphicsResource *outputSpan = nullptr;
			cudaGraphicsResource *outputFillInfo = nullptr;

			cudaGraphicsResource *qm_output_stencil_mask = nullptr;
		} resource;
	} _cuda;

	CUDAArray<int> _qm_mask_table_pixel8;
	CUDAArray<int4> _qm_mask_table_pixel32;

	CUDAArray<float2> _sample_position;
};

} // end of namespace BigFragAM

} // end of namespace Mochimazui
