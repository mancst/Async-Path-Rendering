
#define _CRT_SECURE_NO_WARNINGS

#include "glgui.h"

#include "stdafx.h"


#pragma warning(push)
#pragma warning(disable: 4312)
//#define STB_IMAGE_IMPLEMENTATION
//#define STBI_ONLY_PNG
#include "3rd/stb_image.h"
#pragma warning(pop)

//#define STB_TRUETYPE_IMPLEMENTATION
#include "3rd/stb_truetype.h"

#include <cassert>
#include <ctime>
#include <list>

using namespace Microsoft::WRL;

namespace Mochimazui {
namespace GLGUI {

// -------- -------- -------- -------- -------- -------- -------- --------
UIObject::UIObject() {
	_margin.left = _margin.right = _margin.top = _margin.bottom = 0;
	_border.left = _border.right = _border.top = _border.bottom = 0;
	_padding.left = _padding.right = _padding.top = _padding.bottom = 0;
	_layoutSpace = 2;
}

UIObject::UIObject(const std::weak_ptr<UIObject> &pparent)
	: _obj_parent(pparent) {
	_margin.left = _margin.right = _margin.top = _margin.bottom = 0;
	_border.left = _border.right = _border.top = _border.bottom = 0;
	_padding.left = _padding.right = _padding.top = _padding.bottom = 0;
	_layoutSpace = 2;
}

void UIObject::show(bool f) {
	_show = f;
	auto pp = _obj_parent.lock();
	if (pp) {
		pp->arrangeLayout();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::move(int w, int h) {}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::resize(int w, int h) {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::moveAndResize(int x, int y, int w, int h) {}

// -------- -------- -------- -------- -------- -------- -------- --------
bool UIObject::hitTest(const IVec2 &p) {
	return (_pos.x <= p.x) && (_pos.y <= p.y) &&
		(p.x < (_pos.x + _size.x)) && (p.y < (_pos.y + _size.y));
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::shared_ptr<UIObject> UIObject::findChild(const IVec2 &p) {
	// TODO: improve performance by quad-tree?
	for (auto pc : _obj_children) {
		if (pc->hitTest(p)) { 
			auto pf = pc->findChild(p);
			if (pf) { return pf; }
			return pc;
		}
	}
	return nullptr;
}

// -------- -------- -------- -------- -------- -------- -------- --------
IVec2 UIObject::layoutSize() {
	if (_show) {
		int w = -1;
		int h = -1;
		if (_sizePolicy.x == SP_FIX) {
			w = _size.x + _margin.left + _margin.right;
		}
		if (_sizePolicy.y == SP_FIX) {
			h = _size.y + _margin.top + _margin.bottom;
		}
		return IVec2(w, h);
	} else {
		return IVec2(0, 0);
	}
}

IVec2 UIObject::minSize() {
	int cw = 0;
	int ch = 0;

	if (_obj_id == "ok-cancel") {
		cw = cw;
	}

	for (auto pc : _obj_children) {
		auto ms = pc->minSize();
		cw = std::max(cw, ms.x);
		ch = std::max(ch, ms.y);
	}

	if (_sizePolicy.x == SP_FIX) { cw = std::max(cw, _size.x); }
	if (_sizePolicy.y == SP_FIX) { ch = std::max(ch, _size.y); }

	cw += _border.left + _padding.left + _padding.right + _border.right;
	ch += _border.top + _padding.top + _padding.bottom + _border.bottom;

	return IVec2(cw, ch);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::arrangeLayout() {

	// SizePolicy: fix, auto, min, expand,

	//
	// this-> padding
	// child -> layoutSize
	// [
	//   this->layoutSpace
	//   child2 -> layoutSize
	// ] *
	// this-> padding

	// (layoutSize == -1) -> auto.
	//

	if (_layout == WL_VERTICAL || _layout == WL_HORIZONTAL) {

		std::function<SizePolicy(const SizePolicy2&)> gsp;
		std::function<int(const IVec2&)> gi;
		std::function<int(const ISize2&)> gs;

		if (_layout == WL_HORIZONTAL) {
			gsp = [](const SizePolicy2 &sp) ->SizePolicy {return sp.x; };
			gi = [](const IVec2 &v) -> int {return v.x; };
			gs = [](const ISize2 &v) -> int {return v.w; };
		}
		else {
			gsp = [](const SizePolicy2 &sp) ->SizePolicy {return sp.y; };
			gi = [](const IVec2 &v) -> int {return v.y; };
			gs = [](const ISize2 &v) -> int {return v.h; };
		}

		// ------- -------
		// common part
		auto &cs = _obj_children;

		int autoCount = 0;
		int expandCount = 0;
		int showCount = 0;
		int minCount = 0;
		int fixCount = 0;

		int used = 0;

		std::vector<SizePolicy> csp(cs.size());
		std::vector<int> csize(cs.size());

		// get size policy & count number.
		for (int i = 0; i < cs.size(); ++i) {
			auto pc = cs[i];
			auto sp = gsp(pc->_sizePolicy);
			csp[i] = sp;
			if (pc->_show) {
				++showCount;
				if (sp == SP_FIX) {
					++fixCount;
					csize[i] = gi(pc->size());
				}
				else {
					csize[i] = gi(pc->minSize());
					if (sp == SP_AUTO) { ++autoCount; }
					else if (sp == SP_EXPAND) { ++expandCount; }
					else if (sp == SP_MIN) { ++minCount; }
				}
				used += csize[i];
			}
		}

		//
		if (expandCount) {
			autoCount = 0;
			for (auto &sp : csp) {
				if (sp == SP_AUTO) {
					++minCount;
					sp = SP_MIN;
				}
				else if (sp == SP_EXPAND) {
					++autoCount;
					sp = SP_AUTO;
				}
			}
		}

		//
		int unused = gi(_size)
			- gs(_border.size2() + _padding.size2())
			- used - (showCount - 1) * _layoutSpace;

		for (int i = 0; i < cs.size(); ++i) {
			if (csp[i] == SP_AUTO) {
				auto psize = unused / autoCount;
				csize[i] += psize;
				unused -= psize;
				--autoCount;
			}
		}

		// -------- --------
		if (_layout == WL_HORIZONTAL) {
			int offset = 0;
			offset += _pos.x + _border.left + _padding.left;
			int yy = _pos.y + _border.top + _padding.top;
			int yyy = _size.y - (_border.top + _padding.top + _padding.bottom + _border.bottom);

			for (int i = 0; i < _obj_children.size(); ++i) {
				auto c = _obj_children[i];
				if (!c->_show) { continue; }
				c->_pos.x = offset + c->_margin.left;
				c->_pos.y = yy + c->_margin.top;

				auto new_width = csize[i] - c->_margin.left - c->_margin.right;;
				auto new_height = yyy;
				c->resizeEvent(new_width, new_height);

				offset += csize[i] + _layoutSpace;
			}
		}
		else {
			int offset = 0;
			offset += _pos.y + _border.top + _padding.top;
			int xx = _pos.x + _border.left + _padding.left;
			int xxx = _size.x - (_border.left + _padding.left + _padding.right + _border.right);

			for (int i = 0; i < _obj_children.size(); ++i) {
				auto c = _obj_children[i];
				if (!c->_show) { continue; }

				c->_pos.x = xx + c->_margin.left;
				c->_pos.y = offset + c->_margin.top;

				auto new_width = xxx;
				auto new_height = csize[i] - c->_margin.top - c->_margin.bottom;
				c->resizeEvent(new_width, new_height);

				offset += csize[i] + _layoutSpace;
			}
		}

	}

	for (auto &p : _obj_children) {
		p->arrangeLayout();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::repaint() {
	_repaint = true;
}

void UIObject::repaintEvent() {
	if (_repaint) {
		paintEvent();
		_repaint = false;
		for (auto pc : _obj_children) {
			pc->repaint();
			pc->repaintEvent();
		}
	}
	else {
		for (auto pc : _obj_children) {
			pc->repaintEvent();
		}
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::attributeFromJSON(const nlohmann::json &j) {
	using PRIVATE::dget;

	_obj_id = dget<std::string>(j, "id", "window");

	_pos.x = dget<int>(j, "x", _pos.x);
	_pos.y = dget<int>(j, "y", _pos.y);
	_size.x = dget<int>(j, "width", _size.x);
	_size.y = dget<int>(j, "height", _size.y);

	std::string layout = dget<std::string>(j, "layout", "horizontal");
	if (layout == "horizontal") {
		_layout = WL_HORIZONTAL;
	} else if (layout == "vertical") {
		_layout = WL_VERTICAL;
	} else {
		// ERROR.
	}

	auto getSizePolicy = [&](const std::string &sps, SizePolicy def) {
		if (sps == "auto") { return SP_AUTO; }
		else if (sps == "fix") { return SP_FIX; }
		else if (sps == "min") { return SP_MIN; }
		else if (sps == "expand") { return SP_EXPAND; }
		else { return def; }
	};

	_sizePolicy.x = getSizePolicy(dget<std::string>(j, "size-policy-x"), _sizePolicy.x);
	_sizePolicy.y = getSizePolicy(dget<std::string>(j, "size-policy-y"), _sizePolicy.y);
}

std::weak_ptr<UIObject> UIObject::ui_by_id(const std::string &id) {
	if (!id.length()) { return std::weak_ptr<UIObject>(); }
	if (id[0] == '#') {
		// global name
		auto i = _obj_children_map.find(id);
		if (i == _obj_children_map.end()) { 
			return std::weak_ptr<UIObject>(); 
		}
		return i->second;
	} else {
		// hierarchical name
		auto dotPos = id.find('.');
		if (dotPos == std::string::npos) {
			auto i = _obj_children_map.find(id);
			if (i == _obj_children_map.end()) { return std::weak_ptr<UIObject>(); }
			return i->second;
		} else {
			std::string id0 = id.substr(0, dotPos);
			std::string id1 = id.substr(dotPos + 1);
			auto i = _obj_children_map.find(id0);
			if (i == _obj_children_map.end()) { return std::weak_ptr<UIObject>(); }

			//return i->second->
			return std::weak_ptr<UIObject>();
		}
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::paintChildren() {
	for (auto &p : _obj_children) {
		p->paintEvent();
	}
}

void UIObject::idleEvent() {
	if (_onIdle) { _onIdle(); }
}

void UIObject::paintEvent() {
	if (_onPaint) {
		_onPaint();
	}
	paintChildren();
}

void UIObject::resizeEvent(int w, int h) {
	_size.x = w;
	_size.y = h;
	if (_onResize) {
		_onResize(w, h);
	}
	arrangeLayout();
}

void UIObject::mouseEnterEvent() {
	if (_onMouseEnter) { _onMouseEnter(); }
}

void UIObject::mouseLeaveEvent() {
	if (_onMouseLeave) { _onMouseLeave(); }
}

void UIObject::mouseLeftButtonDownEvent(int x, int y) {
	if (_onMouseLeftButtonDown) { _onMouseLeftButtonDown(x, y); }
}

void UIObject::mouseLeftButtonUpEvent(int x, int y) {
	if (_onMouseLeftButtonUp) { _onMouseLeftButtonUp(x, y); }
}

void UIObject::mouseMiddleButtonDownEvent(int x, int y) {
	if (_onMouseMiddleButtonDown) { _onMouseMiddleButtonDown(x, y); }
}

void UIObject::mouseMiddleButtonUpEvent(int x, int y) {
	if (_onMouseMiddleButtonUp) { _onMouseMiddleButtonUp(x, y); }
}

void UIObject::mouseRightButtonDownEvent(int x, int y) {
	if (_onMouseRightButtonDown) { _onMouseRightButtonDown(x, y); }
}

void UIObject::mouseRightButtonUpEvent(int x, int y) {
	if (_onMouseRightButtonUp) { _onMouseRightButtonUp(x, y); }
}

void UIObject::mouseWheelEvent(int x, int y) {
	if (_onMouseWheel) { _onMouseWheel(x, y); }
}

void UIObject::mouseMoveEvent(int x, int y, uint32_t buttonState) {
	if (_onMouseMove) { _onMouseMove(x, y, buttonState); }
}

void UIObject::textInputEvent(const char *text) {
	if (_onTextInput) { _onTextInput(text); }
}

void UIObject::keyboardEvent(uint32_t type, uint8_t state, SDL_Keysym keysym) {
	if (_onKeyboard) {
		_onKeyboard(type, state, keysym);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------

HWND Window::m_hwnd = nullptr;

Window::Window() {}
Window::Window(const std::shared_ptr<WindowManager> &pmanager)
	:_window_manager(pmanager)
{}
Window::~Window() {
	SDL_GL_DeleteContext(_sdlGLContext);
	SDL_DestroyWindow(_sdlWindow);
}

void Window::resize(int w, int h) {
	if (_sdlWindow) {
		SDL_SetWindowSize(_sdlWindow, w, h);
	}
}

int Window::Run(DXAsset* pAsset, HINSTANCE hInstance, int nCmdShow)
{
	// Parse the command line parameters
	int argc;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	pAsset->ParseCommandLineArgs(argv, argc);
	LocalFree(argv);

	// Initialize the window class.
	WNDCLASSEX windowClass = { 0 };
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = WindowProc;
	windowClass.hInstance = hInstance;
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.lpszClassName = L"DXAssetClass";
	RegisterClassEx(&windowClass);

	RECT windowRect = { 0, 0, static_cast<LONG>(pAsset->GetWidth()), static_cast<LONG>(pAsset->GetHeight()) };
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

	// Create the window and store a handle to it.
	m_hwnd = CreateWindow(
		windowClass.lpszClassName,
		pAsset->GetTitle(),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr,        // We have no parent window.
		nullptr,        // We aren't using menus.
		hInstance,
		pAsset);

	// Initialize the sample. OnInit is defined in each child-implementation of DXAsset.
	pAsset->OnInit();

	//ShowWindow(m_hwnd, nCmdShow);

	// Main sample loop.
	MSG msg = {};
	while (msg.message != WM_QUIT)
	{
		// Process any messages in the queue.
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	pAsset->OnDestroy();

	// Return this part of the WM_QUIT message to Windows.
	return static_cast<char>(msg.wParam);
}

// Main message handler for the sample.
LRESULT CALLBACK Window::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	DXAsset* pAsset = reinterpret_cast<DXAsset*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (message)
	{
	case WM_CREATE:
	{
		// Save the DXAsset* passed in to CreateWindow.
		//LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		//SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
	}
	return 0;

	case WM_KEYDOWN:
		if (pAsset)
		{
			//pSample->OnKeyDown(static_cast<UINT8>(wParam));
		}
		return 0;

	case WM_KEYUP:
		if (pAsset)
		{
			//pSample->OnKeyUp(static_cast<UINT8>(wParam));
		}
		return 0;

	case WM_PAINT:
		if (pAsset)
		{
			//pSample->OnUpdate();
			pAsset->OnRender();
		}
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}

	// Handle any messages the switch statement didn't.
	return DefWindowProc(hWnd, message, wParam, lParam);
}


void Window::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_title = dget<std::string>(j, "title", "GLGUI");
}

void Window::arrangeLayout() {
	for (auto psw : _obj_children) {
		psw->resizeEvent(_size.x, _size.y);
	}
}

void Window::resizeEvent(int w, int h) {
	_size = IVec2(w, h);
	for (auto psw : _obj_children) {
		psw->resizeEvent(_size.x, _size.y);
	}
	repaint();
}

void Window::mouseLeftButtonDownEvent(int x, int y) {
	if (_currentChild) {
		auto p = _currentChild->position();
		_currentChild->mouseLeftButtonDownEvent(x - p.x, y - p.y);
	}
}

void Window::mouseLeftButtonUpEvent(int x, int y) {
	if (_currentChild) {
		auto p = _currentChild->position();
		_currentChild->mouseLeftButtonUpEvent(x - p.x, y - p.y);
	}
}

void Window::mouseMoveEvent(int x, int y, uint32_t buttonState) {
	auto c = findChild(IVec2(x, y));
	auto e = [&]() {
		c->mouseEnterEvent();
		auto p = c->position();
		c->mouseMoveEvent(x - p.x, y - p.y, buttonState);
	};
	if (!c) {
		if (_currentChild) { e(); }
	} else if (c == _currentChild) {
		if (_currentChild) { e(); }
	} else {
		if (_currentChild) { _currentChild->mouseLeaveEvent(); }
		e();
		_currentChild = c;
	}
}

void Window::mouseWheelEvent(int x, int y) {
	if (_currentChild) {
		_currentChild->mouseWheelEvent(x, y);
	}
}

void Window::textInputEvent(const char *text) {
	if (_currentChild) {
		_currentChild->textInputEvent(text);
	}
}

void Window::keyboardEvent(uint32_t type, uint8_t state, SDL_Keysym keysym) {
	if (_currentChild) {
		_currentChild->keyboardEvent(type, state, keysym);
	}
}

std::shared_ptr<SharedPaintData> Window::sharedPaintData() {
	if (!_sharedPaintData) {
		_sharedPaintData.reset(new SharedPaintData);
		_sharedPaintData->init();
	}
	return _sharedPaintData;
}

void Window::repaintEvent() {
	paintEvent();
}

std::weak_ptr<UIObject> Window::ui_by_id(const std::string &id) {
	if (_obj_children.size()) {
		return _obj_children[0]->ui_by_id(id);
	}
	else {
		return std::weak_ptr<UIObject>();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::shared_ptr<Window> Window::createWindowFromJSON(const std::string &json_str) {
	return createWindowFromJSON(json_str, nullptr);
}

std::shared_ptr<Window> Window::createWindowFromJSONFile(const std::string &fileName) {
	return createWindowFromJSONFile(fileName, nullptr);
}

std::shared_ptr<Window> Window::createWindowFromJSON(const std::string &json_str,
	std::function<void(std::shared_ptr<Window>&pw)> onCreate) {

	using nlohmann::json;
	using PRIVATE::dget;

	json j;
	try {
		j = json::parse(json_str);
	}
	catch (std::exception &e) {
		printf("In createUi: %s\n", e.what());
		return nullptr;
	}

	auto w = j["window"];
	if (w.is_null()) {
		printf("createUi: Warning: \"window\" not found.\n");
		return nullptr;
	}

	std::shared_ptr<Window> pWindow(new Window());
	pWindow->attributeFromJSON(w);
	pWindow->createSDLWindow();

	// create subwindow.
	auto subWindowList = w["subwindows"];

	if (!subWindowList.is_null()) {

		auto subWindowRefList = j["subwindows"];

		// create an empty agency.
		std::shared_ptr<SubWindow> prsw(new SubWindow(pWindow));
		prsw->_obj_id = pWindow->_obj_id;
		prsw->_size = pWindow->_size;
		prsw->_layout = pWindow->_layout;
		prsw->_border = ISize4(0);
		pWindow->_obj_children.push_back(prsw);

		std::function<void(const json &list, std::shared_ptr<SubWindow> &pp)> createSubWindow;
		
		createSubWindow = [&createSubWindow, &subWindowRefList, &pWindow]
			(const json &list, std::shared_ptr<SubWindow> &pp) {

			//
			for (auto &swj : list) {
				std::string type = dget<std::string>(swj, "type", "");
				std::shared_ptr < SubWindow > psw; 

				if (type.empty()) { psw.reset(new SubWindow(pp)); }
				else if (type == "subwindow") { psw.reset(new SubWindow(pp)); }
				else if (type == "scroll-window") { psw.reset(new ScrollWindow(pp)); }
				else if (type == "horizontal-layout" || type == "hlayout") { psw.reset(new HorizontalLayout(pp)); }
				else if (type == "vertical-layout" || type == "vlayout") { psw.reset(new VerticalLayout(pp)); }
				else if (type == "horizontal-spacer") { psw.reset(new HorizontalSpacer(pp)); }
				else if (type == "vertical-spacer") { psw.reset(new VerticalSpacer(pp)); }
				else if (type == "frame") { psw.reset(new Frame(pp)); }
				else if (type == "horizontal-line") { psw.reset(new HorizontalLine(pp)); }
				else if (type == "vertical-line") { psw.reset(new VerticalLine(pp)); }
				else if (type == "label") { psw.reset(new Label(pp)); }
				else if (type == "push-button") { psw.reset(new PushButton(pp)); }
				else if (type == "radio-button") { psw.reset(new RadioButton(pp)); }
				else if (type == "check-box") { psw.reset(new CheckBox(pp)); }
				else if (type == "horizontal-slider") { psw.reset(new HorizontalSlider(pp)); }
				else {
					printf("In createWindowFromJSON:\n\tError: illegal type\"%s\".\n", type.c_str());
					psw.reset(new SubWindow(pp));
				}
				psw->attributeFromJSON(swj);

				//
				// id: ascii string without '.'
				//   start with '#' : global name.
				//   else: local name.
				//     
				auto &id = psw->id();
				if (id.empty()) {
					printf("In createWindowFromJSON: subwindow has empty id.\n");
					continue;
				}
				if (id[0] == '#') {
					// global name
					auto &cmap = pWindow->_obj_children_map;
					auto i = cmap.find(id);
					if (i != cmap.end()) {
						printf("In createWindowFromJSON: id \"%s\" already used.\n", id.c_str());
						continue;
					}
					cmap[id] = psw;
				}
				else {
					// local name
					auto &cmap = pp->_obj_children_map;
					auto i = cmap.find(id);
					if (i != cmap.end()) {
						printf("In createWindowFromJSON: id \"%s\" already used.\n", id.c_str());
						continue;
					}
					cmap[id] = psw;
				}
				pp->_obj_children.push_back(psw);

				auto subWindowList = swj["subwindows"];
				if (!subWindowList.is_null()) {
					createSubWindow(subWindowList, psw);
				}

			}
		};
		createSubWindow(subWindowList, prsw);
	}
	if (onCreate) {
		onCreate(pWindow);
	}
	pWindow->arrangeLayout();
	pWindow->repaint();
	return pWindow;
}

std::shared_ptr<Window> Window::createWindowFromJSONFile(const std::string &fileName,
	std::function<void(std::shared_ptr<Window>&pw)> onCreate) {

	std::shared_ptr<Window> pWindow;
	char *json_text = nullptr;

	FILE *fin;
	fin = fopen(fileName.c_str(), "rb");
	if (!fin) {
		printf("createWindowFromJSONFile: cannot open input file.");
		return nullptr;
	}

	fseek(fin, 0, SEEK_END);
	auto size = ftell(fin);
	json_text = new char[size + 1];
	if (!json_text) {
		printf("createWindowFromJSONFile: cannot open input file.");
		return nullptr;
	}
	fseek(fin, 0, SEEK_SET);
	size_t size_read = fread(json_text, 1, size, fin);
	json_text[size] = '\0';

	try {
		pWindow = Window::createWindowFromJSON(json_text, onCreate);
	}
	catch (std::exception &e) {
		printf("In createUiFromFile: %s\n", e.what());
	}

	delete[] json_text;

	return pWindow;
}

void Window::idleEvent() {
	UIObject::idleEvent();
}

void Window::paintEvent() {

	GLGUI_CHECK_GL_ERROR();

	// count fps & set to window title.
	//static std::list<int> timestamps;
	//int now = clock();
	//while (!timestamps.empty() && timestamps.front() + 1000 < now) {
	//	timestamps.pop_front();
	//}
	//timestamps.push_back(now);

	//float fps = 1000.0f / ((timestamps.back() - timestamps.front()) / (float)timestamps.size());
	//char fpss[128];
	//sprintf(fpss, "Frame per second: %.2f, Time per frame: %.2f", fps, 1000.f / std::max(1.f, fps));
	//auto newTitle = _title + " VG: " + fpss;
	//SDL_SetWindowTitle(_sdlWindow, newTitle.c_str());

	// paint.

	//glDisable(GL_SCISSOR_TEST);
	//glViewport(0, 0, _size.x, _size.y);
	//glClearColor(.25f, .25f, .25f, 1.f);
	//glClear(GL_COLOR_BUFFER_BIT);

	UIObject::paintEvent();
	SDL_GL_SwapWindow(_sdlWindow);

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void Window::createSDLWindow() {

	// Create the window and store a handle to it.
	_sdlWindow = SDL_CreateWindow(_title.c_str(), 
		//SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, 32,
		_size.x, _size.y, 
		//SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
		SDL_WINDOW_OPENGL);

	if (!_sdlWindow) {
		printf("Window::createSDLWindow: %s\n", SDL_GetError());
	}

	_sdlGLContext = SDL_GL_CreateContext(_sdlWindow);
	if (!_sdlGLContext) {
		printf("Window::createSDLWindow: %s\n", SDL_GetError());
	}

	if (ogl_LoadFunctions() == ogl_LOAD_FAILED) {
		printf("ogl_LoadFunctions Error.\n");
	}
	//GLGUI_CHECK_GL_ERROR();
	//auto e = glewInit();
	//GLGUI_CHECK_GL_ERROR();
	//if (e != GLEW_OK) {
	//	printf("GLEW ERROR.");
	//}

	//SDL_SetWindowFullscreen(_sdlWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);

	SDL_GL_SetSwapInterval(1);
}

// -------- -------- -------- -------- -------- -------- -------- --------
bool Window::eventHandle(const SDL_Event &e) {

	switch (e.type) {

	case SDL_WINDOWEVENT:
		switch (e.window.event) {
		case SDL_WINDOWEVENT_RESIZED:
		case SDL_WINDOWEVENT_SIZE_CHANGED:
			resizeEvent(e.window.data1, e.window.data2);
			return true;
		default:
			break;
		}
		break;

	case SDL_MOUSEBUTTONDOWN:
		if (!_currentChild) { break; }
		switch (e.button.button)
		{
		case SDL_BUTTON_LEFT:
			mouseLeftButtonDownEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_MIDDLE:
			_currentChild->mouseMiddleButtonDownEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_RIGHT:
			_currentChild->mouseRightButtonDownEvent(e.button.x, e.button.y);
			return true;
		default:
			break;
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (!_currentChild) { break; }
		switch (e.button.button)
		{
		case SDL_BUTTON_LEFT:
			mouseLeftButtonUpEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_MIDDLE:
			_currentChild->mouseMiddleButtonUpEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_RIGHT:
			_currentChild->mouseRightButtonUpEvent(e.button.x, e.button.y);
			return true;
		default:
			break;
		}
		break;
	case SDL_MOUSEMOTION:
		mouseMoveEvent(e.motion.x, e.motion.y, e.motion.state);
		break;
	case SDL_MOUSEWHEEL:
		mouseWheelEvent(e.wheel.x, e.wheel.y);
		return true;
		break;
	case SDL_KEYDOWN:
		keyboardEvent(SDL_KEYDOWN, e.key.state, e.key.keysym);
		break;
	case SDL_TEXTINPUT:
		textInputEvent(e.text.text);
		return true;
		break;
	case SDL_QUIT:
		// ignore.
		break;
	}
	return false;
}

// -------- -------- -------- -------- -------- -------- -------- --------

SubWindow::SubWindow(const std::weak_ptr<Window> &pparent)
	:UIObject(pparent), _root_window(pparent)
{}

SubWindow::SubWindow(const std::weak_ptr<SubWindow> &pparent)
	:UIObject(pparent), _root_window(pparent.lock()->_root_window)
{}

std::weak_ptr<PaintData> SubWindow::paintData() {
	if (!_paintData) { _paintData.reset(new PaintData); }
	return _paintData;
}

//void SubWindow::paintEvent() {
//	paintChildren();
//}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Window manager

// -------- -------- -------- -------- -------- -------- -------- --------
void WindowManager::addWindow(std::shared_ptr<Window> &pw) {
	if (pw) { _windows.push_back(pw); }
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::weak_ptr<Window> WindowManager::createWindow() {
	return createWindowFromJSON("{}");
}

std::weak_ptr<Window> WindowManager::createWindowFromJSON(const std::string &json_str) {
	auto pw = Window::createWindowFromJSON(json_str);
	addWindow(pw);
	return pw;
}

std::weak_ptr<Window> WindowManager::createWindowFromJSONFile(const std::string &fileName) {
	auto pw = Window::createWindowFromJSONFile(fileName);
	addWindow(pw);
	return pw;
}

void WindowManager::repaintEvent() {
	for (auto pw : _windows) {
		pw->repaintEvent();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void WindowManager::idleEvent() {
	for (auto &pw : _windows) {
		pw->idleEvent();
	}
}

bool WindowManager::eventHandle(const SDL_Event &e) {
	// TODO: dispatch event by window id.
	if (_windows.size()) {
		return _windows[0]->eventHandle(e);
	}
	return false;
}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// DXAsset

// -------- -------- -------- -------- -------- -------- -------- --------

// InterlockedCompareExchange returns the object's value if the 
// comparison fails.  If it is already 0, then its value won't 
// change and 0 will be returned.
#define InterlockedGetValue(object) InterlockedCompareExchange(object, 0, 0)

const float DXAsset::ParticleSpread = 400.0f;

DXAsset::DXAsset()
{
}

DXAsset::DXAsset(UINT width, UINT height, std::wstring name) :
	m_width(width),
	m_height(height),
	m_title(name),
	m_useWarpDevice(false),
	m_frameIndex(0),
	m_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)),
	m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
	m_rtvDescriptorSize(0),
	m_srvUavDescriptorSize(0),
	m_pConstantBufferGSData(nullptr),
	m_renderContextFenceValue(0),
	m_terminating(0),
	m_srvIndex{},
	m_frameFenceValues{}

{
	WCHAR assetsPath[512];
	GetAssetsPath(assetsPath, _countof(assetsPath));
	m_assetsPath = assetsPath;

	m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);

	for (int n = 0; n < ThreadCount; n++)
	{
		m_renderContextFenceValues[n] = 0;
		m_threadFenceValues[n] = 0;
	}

	float sqRootNumAsyncContexts = sqrt(static_cast<float>(ThreadCount));
	m_heightInstances = static_cast<UINT>(ceil(sqRootNumAsyncContexts));
	m_widthInstances = static_cast<UINT>(ceil(sqRootNumAsyncContexts));

	if (m_widthInstances * (m_heightInstances - 1) >= ThreadCount)
	{
		m_heightInstances--;
	}
}

DXAsset::~DXAsset()
{
}

void DXAsset::OnInit()
{

	LoadPipeline();
	LoadAssets();
	CreateAsyncContexts();
}

// Load the rendering pipeline dependencies.
void DXAsset::LoadPipeline()
{
	UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
	// Enable the debug layer (requires the Graphics Tools "optional feature").
	// NOTE: Enabling the debug layer after device creation will invalidate the active device.
	{
		ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
		{
			debugController->EnableDebugLayer();

			// Enable additional debug layers.
			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
	}
#endif

	ComPtr<IDXGIFactory4> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	if (m_useWarpDevice)
	{
		ComPtr<IDXGIAdapter> warpAdapter;
		ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

		ThrowIfFailed(D3D12CreateDevice(
			warpAdapter.Get(),
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&m_device)
		));
	}
	else
	{
		ComPtr<IDXGIAdapter1> hardwareAdapter;
		GetHardwareAdapter(factory.Get(), &hardwareAdapter);

		ThrowIfFailed(D3D12CreateDevice(
			hardwareAdapter.Get(),
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&m_device)
		));
	}

	// Describe and create the command queue.
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));
	NAME_D3D12_OBJECT(m_commandQueue);

	// Describe and create the swap chain.
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = FrameCount;
	swapChainDesc.Width = m_width;
	swapChainDesc.Height = m_height;
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

	ComPtr<IDXGISwapChain1> swapChain;
	ThrowIfFailed(factory->CreateSwapChainForHwnd(
		m_commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
		Win32Application::GetHwnd(),
		&swapChainDesc,
		nullptr,
		nullptr,
		&swapChain
	));

	// This sample does not support fullscreen transitions.
	ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain.As(&m_swapChain));
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	m_swapChainEvent = m_swapChain->GetFrameLatencyWaitableObject();

	// Create descriptor heaps.
	{
		// Describe and create a render target view (RTV) descriptor heap.
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = FrameCount;
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

		// Describe and create a shader resource view (SRV) and unordered
		// access view (UAV) descriptor heap.
		D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
		srvUavHeapDesc.NumDescriptors = DescriptorCount;
		srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		ThrowIfFailed(m_device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&m_srvUavHeap)));
		NAME_D3D12_OBJECT(m_srvUavHeap);

		m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		m_srvUavDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// Create frame resources.
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

		// Create a RTV and a command allocator for each frame.
		for (UINT n = 0; n < FrameCount; n++)
		{
			ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
			m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
			rtvHandle.Offset(1, m_rtvDescriptorSize);

			NAME_D3D12_OBJECT_INDEXED(m_renderTargets, n);

			ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[n])));
		}
	}
}

// Load the sample assets.
void DXAsset::LoadAssets()
{
	// Create the root signatures.
	{
		D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

		// This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
		featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

		if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
		{
			featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
		}

		// Graphics root signature.
		{
			CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

			CD3DX12_ROOT_PARAMETER1 rootParameters[GraphicsRootParametersCount];
			rootParameters[GraphicsRootCBV].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL);
			rootParameters[GraphicsRootSRVTable].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_VERTEX);

			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
			rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, featureData.HighestVersion, &signature, &error));
			ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
			NAME_D3D12_OBJECT(m_rootSignature);
		}

		// Compute root signature.
		{
			CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);

			CD3DX12_ROOT_PARAMETER1 rootParameters[ComputeRootParametersCount];
			rootParameters[ComputeRootCBV].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL);
			rootParameters[ComputeRootSRVTable].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
			rootParameters[ComputeRootUAVTable].InitAsDescriptorTable(1, &ranges[1], D3D12_SHADER_VISIBILITY_ALL);

			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
			computeRootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr);

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, featureData.HighestVersion, &signature, &error));
			ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));
			NAME_D3D12_OBJECT(m_computeRootSignature);
		}
	}

	// Create the pipeline states, which includes compiling and loading shaders.
	{
		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> geometryShader;
		ComPtr<ID3DBlob> pixelShader;
		ComPtr<ID3DBlob> computeShader;

#if defined(_DEBUG)
		// Enable better shader debugging with the graphics debugging tools.
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
		UINT compileFlags = 0;
#endif

		// Load and compile shaders.
		ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"ParticleDraw.hlsl").c_str(), nullptr, nullptr, "VSParticleDraw", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
		ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"ParticleDraw.hlsl").c_str(), nullptr, nullptr, "GSParticleDraw", "gs_5_0", compileFlags, 0, &geometryShader, nullptr));
		ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"ParticleDraw.hlsl").c_str(), nullptr, nullptr, "PSParticleDraw", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));
		ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"NBodyGravityCS.hlsl").c_str(), nullptr, nullptr, "CSMain", "cs_5_0", compileFlags, 0, &computeShader, nullptr));

		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		};

		// Describe the blend and depth states.
		CD3DX12_BLEND_DESC blendDesc(D3D12_DEFAULT);
		blendDesc.RenderTarget[0].BlendEnable = TRUE;
		blendDesc.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
		blendDesc.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
		blendDesc.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ZERO;
		blendDesc.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;

		CD3DX12_DEPTH_STENCIL_DESC depthStencilDesc(D3D12_DEFAULT);
		depthStencilDesc.DepthEnable = FALSE;
		depthStencilDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;

		// Describe and create the graphics pipeline state object (PSO).
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
		psoDesc.pRootSignature = m_rootSignature.Get();
		psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
		psoDesc.GS = CD3DX12_SHADER_BYTECODE(geometryShader.Get());
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psoDesc.BlendState = blendDesc;
		psoDesc.DepthStencilState = depthStencilDesc;
		psoDesc.SampleMask = UINT_MAX;
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
		psoDesc.NumRenderTargets = 1;
		psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
		psoDesc.SampleDesc.Count = 1;

		ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));
		NAME_D3D12_OBJECT(m_pipelineState);

		// Describe and create the compute pipeline state object (PSO).
		D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
		computePsoDesc.pRootSignature = m_computeRootSignature.Get();
		computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

		ThrowIfFailed(m_device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_computeState)));
		NAME_D3D12_OBJECT(m_computeState);
	}

	// Create the command list.
	ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList)));
	NAME_D3D12_OBJECT(m_commandList);

	CreateVertexBuffer();
	CreateParticleBuffers();

	// Note: ComPtr's are CPU objects but this resource needs to stay in scope until
	// the command list that references it has finished executing on the GPU.
	// We will flush the GPU at the end of this method to ensure the resource is not
	// prematurely destroyed.
	ComPtr<ID3D12Resource> constantBufferCSUpload;

	// Create the compute shader's constant buffer.
	{
		const UINT bufferSize = sizeof(ConstantBufferCS);

		ThrowIfFailed(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&m_constantBufferCS)));

		ThrowIfFailed(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&constantBufferCSUpload)));

		NAME_D3D12_OBJECT(m_constantBufferCS);

		ConstantBufferCS constantBufferCS = {};
		constantBufferCS.param[0] = ParticleCount;
		constantBufferCS.param[1] = int(ceil(ParticleCount / 128.0f));
		constantBufferCS.paramf[0] = 0.1f;
		constantBufferCS.paramf[1] = 1.0f;

		D3D12_SUBRESOURCE_DATA computeCBData = {};
		computeCBData.pData = reinterpret_cast<UINT8*>(&constantBufferCS);
		computeCBData.RowPitch = bufferSize;
		computeCBData.SlicePitch = computeCBData.RowPitch;

		UpdateSubresources<1>(m_commandList.Get(), m_constantBufferCS.Get(), constantBufferCSUpload.Get(), 0, 0, 1, &computeCBData);
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_constantBufferCS.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));
	}

	// Create the geometry shader's constant buffer.
	{
		const UINT constantBufferGSSize = sizeof(ConstantBufferGS) * FrameCount;

		ThrowIfFailed(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(constantBufferGSSize),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_constantBufferGS)
		));

		NAME_D3D12_OBJECT(m_constantBufferGS);

		CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
		ThrowIfFailed(m_constantBufferGS->Map(0, &readRange, reinterpret_cast<void**>(&m_pConstantBufferGSData)));
		ZeroMemory(m_pConstantBufferGSData, constantBufferGSSize);
	}

	// Close the command list and execute it to begin the initial GPU setup.
	ThrowIfFailed(m_commandList->Close());
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	// Create synchronization objects and wait until assets have been uploaded to the GPU.
	{
		ThrowIfFailed(m_device->CreateFence(m_renderContextFenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_renderContextFence)));
		m_renderContextFenceValue++;

		m_renderContextFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (m_renderContextFenceEvent == nullptr)
		{
			ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
		}

		WaitForRenderContext();
	}
}

// Create the particle vertex buffer.
void DXAsset::CreateVertexBuffer()
{
	std::vector<ParticleVertex> vertices;
	vertices.resize(ParticleCount);
	for (UINT i = 0; i < ParticleCount; i++)
	{
		vertices[i].color = XMFLOAT4(1.0f, 1.0f, 0.2f, 1.0f);
	}
	const UINT bufferSize = ParticleCount * sizeof(ParticleVertex);

	ThrowIfFailed(m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_vertexBuffer)));

	ThrowIfFailed(m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_vertexBufferUpload)));

	NAME_D3D12_OBJECT(m_vertexBuffer);

	D3D12_SUBRESOURCE_DATA vertexData = {};
	vertexData.pData = reinterpret_cast<UINT8*>(&vertices[0]);
	vertexData.RowPitch = bufferSize;
	vertexData.SlicePitch = vertexData.RowPitch;

	UpdateSubresources<1>(m_commandList.Get(), m_vertexBuffer.Get(), m_vertexBufferUpload.Get(), 0, 0, 1, &vertexData);
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_vertexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

	m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
	m_vertexBufferView.SizeInBytes = static_cast<UINT>(bufferSize);
	m_vertexBufferView.StrideInBytes = sizeof(ParticleVertex);
}

// Random percent value, from -1 to 1.
float DXAsset::RandomPercent()
{
	float ret = static_cast<float>((rand() % 10000) - 5000);
	return ret / 5000.0f;
}

void DXAsset::LoadParticles(_Out_writes_(numParticles) Particle* pParticles, const XMFLOAT3& center, const XMFLOAT4& velocity, float spread, UINT numParticles)
{
	srand(0);
	for (UINT i = 0; i < numParticles; i++)
	{
		XMFLOAT3 delta(spread, spread, spread);

		while (XMVectorGetX(XMVector3LengthSq(XMLoadFloat3(&delta))) > spread * spread)
		{
			delta.x = RandomPercent() * spread;
			delta.y = RandomPercent() * spread;
			delta.z = RandomPercent() * spread;
		}

		pParticles[i].position.x = center.x + delta.x;
		pParticles[i].position.y = center.y + delta.y;
		pParticles[i].position.z = center.z + delta.z;
		pParticles[i].position.w = 10000.0f * 10000.0f;

		pParticles[i].velocity = velocity;
	}
}

// Create the position and velocity buffer shader resources.
void DXAsset::CreateParticleBuffers()
{
	// Initialize the data in the buffers.
	std::vector<Particle> data;
	data.resize(ParticleCount);
	const UINT dataSize = ParticleCount * sizeof(Particle);

	// Split the particles into two groups.
	float centerSpread = ParticleSpread * 0.50f;
	LoadParticles(&data[0], XMFLOAT3(centerSpread, 0, 0), XMFLOAT4(0, 0, -20, 1 / 100000000.0f), ParticleSpread, ParticleCount / 2);
	LoadParticles(&data[ParticleCount / 2], XMFLOAT3(-centerSpread, 0, 0), XMFLOAT4(0, 0, 20, 1 / 100000000.0f), ParticleSpread, ParticleCount / 2);

	D3D12_HEAP_PROPERTIES defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	D3D12_HEAP_PROPERTIES uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(dataSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	D3D12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(dataSize);

	for (UINT index = 0; index < ThreadCount; index++)
	{
		// Create two buffers in the GPU, each with a copy of the particles data.
		// The compute shader will update one of them while the rendering thread 
		// renders the other. When rendering completes, the threads will swap 
		// which buffer they work on.

		ThrowIfFailed(m_device->CreateCommittedResource(
			&defaultHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&m_particleBuffer0[index])));

		ThrowIfFailed(m_device->CreateCommittedResource(
			&defaultHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&m_particleBuffer1[index])));

		ThrowIfFailed(m_device->CreateCommittedResource(
			&uploadHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&uploadBufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_particleBuffer0Upload[index])));

		ThrowIfFailed(m_device->CreateCommittedResource(
			&uploadHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&uploadBufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_particleBuffer1Upload[index])));

		NAME_D3D12_OBJECT_INDEXED(m_particleBuffer0, index);
		NAME_D3D12_OBJECT_INDEXED(m_particleBuffer1, index);

		D3D12_SUBRESOURCE_DATA particleData = {};
		particleData.pData = reinterpret_cast<UINT8*>(&data[0]);
		particleData.RowPitch = dataSize;
		particleData.SlicePitch = particleData.RowPitch;

		UpdateSubresources<1>(m_commandList.Get(), m_particleBuffer0[index].Get(), m_particleBuffer0Upload[index].Get(), 0, 0, 1, &particleData);
		UpdateSubresources<1>(m_commandList.Get(), m_particleBuffer1[index].Get(), m_particleBuffer1Upload[index].Get(), 0, 0, 1, &particleData);
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_particleBuffer0[index].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_particleBuffer1[index].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));

		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Buffer.FirstElement = 0;
		srvDesc.Buffer.NumElements = ParticleCount;
		srvDesc.Buffer.StructureByteStride = sizeof(Particle);
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

		CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle0(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), SrvParticlePosVelo0 + index, m_srvUavDescriptorSize);
		CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle1(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), SrvParticlePosVelo1 + index, m_srvUavDescriptorSize);
		m_device->CreateShaderResourceView(m_particleBuffer0[index].Get(), &srvDesc, srvHandle0);
		m_device->CreateShaderResourceView(m_particleBuffer1[index].Get(), &srvDesc, srvHandle1);

		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.FirstElement = 0;
		uavDesc.Buffer.NumElements = ParticleCount;
		uavDesc.Buffer.StructureByteStride = sizeof(Particle);
		uavDesc.Buffer.CounterOffsetInBytes = 0;
		uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

		CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle0(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), UavParticlePosVelo0 + index, m_srvUavDescriptorSize);
		CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle1(m_srvUavHeap->GetCPUDescriptorHandleForHeapStart(), UavParticlePosVelo1 + index, m_srvUavDescriptorSize);
		m_device->CreateUnorderedAccessView(m_particleBuffer0[index].Get(), nullptr, &uavDesc, uavHandle0);
		m_device->CreateUnorderedAccessView(m_particleBuffer1[index].Get(), nullptr, &uavDesc, uavHandle1);
	}
}

void DXAsset::CreateAsyncContexts()
{
	for (UINT threadIndex = 0; threadIndex < ThreadCount; ++threadIndex)
	{
		// Create compute resources.
		D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE };
		ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_computeCommandQueue[threadIndex])));
		ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_computeAllocator[threadIndex])));
		ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_computeAllocator[threadIndex].Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList[threadIndex])));
		ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_threadFences[threadIndex])));

		m_threadFenceEvents[threadIndex] = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (m_threadFenceEvents[threadIndex] == nullptr)
		{
			ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
		}

		m_threadData[threadIndex].pContext = this;
		m_threadData[threadIndex].threadIndex = threadIndex;

		m_threadHandles[threadIndex] = CreateThread(
			nullptr,
			0,
			reinterpret_cast<LPTHREAD_START_ROUTINE>(ThreadProc),
			reinterpret_cast<void*>(&m_threadData[threadIndex]),
			CREATE_SUSPENDED,
			nullptr);

		ResumeThread(m_threadHandles[threadIndex]);
	}
}

// Update frame-based values.
void DXAsset::OnUpdate()
{
	// Wait for the previous Present to complete.
	WaitForSingleObjectEx(m_swapChainEvent, 100, FALSE);

	m_timer.Tick(NULL);

	ConstantBufferGS constantBufferGS = {};

	UINT8* destination = m_pConstantBufferGSData + sizeof(ConstantBufferGS) * m_frameIndex;
	memcpy(destination, &constantBufferGS, sizeof(ConstantBufferGS));
}

// Render the scene.
void DXAsset::OnRender()
{
	// Let the compute thread know that a new frame is being rendered.
	for (int n = 0; n < ThreadCount; n++)
	{
		InterlockedExchange(&m_renderContextFenceValues[n], m_renderContextFenceValue);
	}

	// Compute work must be completed before the frame can render or else the SRV 
	// will be in the wrong state.
	for (UINT n = 0; n < ThreadCount; n++)
	{
		UINT64 threadFenceValue = InterlockedGetValue(&m_threadFenceValues[n]);
		if (m_threadFences[n]->GetCompletedValue() < threadFenceValue)
		{
			// Instruct the rendering command queue to wait for the current 
			// compute work to complete.
			ThrowIfFailed(m_commandQueue->Wait(m_threadFences[n].Get(), threadFenceValue));
		}
	}

	PIXBeginEvent(m_commandQueue.Get(), 0, L"Render");

	// Record all the commands we need to render the scene into the command list.
	PopulateCommandList();

	// Execute the command list.
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	PIXEndEvent(m_commandQueue.Get());

	// Present the frame.
	ThrowIfFailed(m_swapChain->Present(1, 0));

	MoveToNextFrame();
}

// Fill the command list with all the render commands and dependent state.
void DXAsset::PopulateCommandList()
{
	// Command list allocators can only be reset when the associated
	// command lists have finished execution on the GPU; apps should use
	// fences to determine GPU execution progress.
	ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

	// However, when ExecuteCommandList() is called on a particular command
	// list, that command list can then be reset at any time and must be before
	// re-recording.
	ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));

	// Set necessary state.
	m_commandList->SetPipelineState(m_pipelineState.Get());
	m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());

	m_commandList->SetGraphicsRootConstantBufferView(GraphicsRootCBV, m_constantBufferGS->GetGPUVirtualAddress() + m_frameIndex * sizeof(ConstantBufferGS));

	ID3D12DescriptorHeap* ppHeaps[] = { m_srvUavHeap.Get() };
	m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

	m_commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
	m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
	m_commandList->RSSetScissorRects(1, &m_scissorRect);

	// Indicate that the back buffer will be used as a render target.
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
	m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

	// Record commands.
	const float clearColor[] = { 0.0f, 0.0f, 0.1f, 0.0f };
	m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

	// Render the particles.
	float viewportHeight = static_cast<float>(static_cast<UINT>(m_viewport.Height) / m_heightInstances);
	float viewportWidth = static_cast<float>(static_cast<UINT>(m_viewport.Width) / m_widthInstances);
	for (UINT n = 0; n < ThreadCount; n++)
	{
		const UINT srvIndex = n + (m_srvIndex[n] == 0 ? SrvParticlePosVelo0 : SrvParticlePosVelo1);

		CD3DX12_VIEWPORT viewport(
			(n % m_widthInstances) * viewportWidth,
			(n / m_widthInstances) * viewportHeight,
			viewportWidth,
			viewportHeight);

		m_commandList->RSSetViewports(1, &viewport);

		CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), srvIndex, m_srvUavDescriptorSize);
		m_commandList->SetGraphicsRootDescriptorTable(GraphicsRootSRVTable, srvHandle);

		PIXBeginEvent(m_commandList.Get(), 0, L"Draw particles for thread %u", n);
		m_commandList->DrawInstanced(ParticleCount, 1, 0, 0);
		PIXEndEvent(m_commandList.Get());
	}

	m_commandList->RSSetViewports(1, &m_viewport);

	// Indicate that the back buffer will now be used to present.
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

	ThrowIfFailed(m_commandList->Close());
}

DWORD DXAsset::AsyncComputeThreadProc(int threadIndex)
{
	ID3D12CommandQueue* pCommandQueue = m_computeCommandQueue[threadIndex].Get();
	ID3D12CommandAllocator* pCommandAllocator = m_computeAllocator[threadIndex].Get();
	ID3D12GraphicsCommandList* pCommandList = m_computeCommandList[threadIndex].Get();
	ID3D12Fence* pFence = m_threadFences[threadIndex].Get();

	while (0 == InterlockedGetValue(&m_terminating))
	{
		// Run the particle simulation.
		Simulate(threadIndex);

		// Close and execute the command list.
		ThrowIfFailed(pCommandList->Close());
		ID3D12CommandList* ppCommandLists[] = { pCommandList };

		PIXBeginEvent(pCommandQueue, 0, L"Thread %d: Iterate on the particle simulation", threadIndex);
		pCommandQueue->ExecuteCommandLists(1, ppCommandLists);
		PIXEndEvent(pCommandQueue);

		// Wait for the compute shader to complete the simulation.
		UINT64 threadFenceValue = InterlockedIncrement(&m_threadFenceValues[threadIndex]);
		ThrowIfFailed(pCommandQueue->Signal(pFence, threadFenceValue));
		ThrowIfFailed(pFence->SetEventOnCompletion(threadFenceValue, m_threadFenceEvents[threadIndex]));
		WaitForSingleObject(m_threadFenceEvents[threadIndex], INFINITE);

		// Wait for the render thread to be done with the SRV so that
		// the next frame in the simulation can run.
		UINT64 renderContextFenceValue = InterlockedGetValue(&m_renderContextFenceValues[threadIndex]);
		if (m_renderContextFence->GetCompletedValue() < renderContextFenceValue)
		{
			ThrowIfFailed(pCommandQueue->Wait(m_renderContextFence.Get(), renderContextFenceValue));
			InterlockedExchange(&m_renderContextFenceValues[threadIndex], 0);
		}

		// Swap the indices to the SRV and UAV.
		m_srvIndex[threadIndex] = 1 - m_srvIndex[threadIndex];

		// Prepare for the next frame.
		ThrowIfFailed(pCommandAllocator->Reset());
		ThrowIfFailed(pCommandList->Reset(pCommandAllocator, m_computeState.Get()));
	}

	return 0;
}

// Run the particle simulation using the compute shader.
void DXAsset::Simulate(UINT threadIndex)
{
	ID3D12GraphicsCommandList* pCommandList = m_computeCommandList[threadIndex].Get();

	UINT srvIndex;
	UINT uavIndex;
	ID3D12Resource *pUavResource;
	if (m_srvIndex[threadIndex] == 0)
	{
		srvIndex = SrvParticlePosVelo0;
		uavIndex = UavParticlePosVelo1;
		pUavResource = m_particleBuffer1[threadIndex].Get();
	}
	else
	{
		srvIndex = SrvParticlePosVelo1;
		uavIndex = UavParticlePosVelo0;
		pUavResource = m_particleBuffer0[threadIndex].Get();
	}

	pCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pUavResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

	//Initialize asynchronous pipeline order.
	//The execution sequence of the pipeline batches are the same as the data arrangement.
	pCommandList->SetPipelineState(m_computeState.Get());
	pCommandList->SetComputeRootSignature(m_computeRootSignature.Get());

	//Each curve subdivider thread must be initialized with the graphic primitive attributes.
	ID3D12DescriptorHeap* ppHeaps[] = { m_srvUavHeap.Get() };
	pCommandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

	CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), srvIndex + threadIndex, m_srvUavDescriptorSize);
	CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), uavIndex + threadIndex, m_srvUavDescriptorSize);

	pCommandList->SetComputeRootConstantBufferView(ComputeRootCBV, m_constantBufferCS->GetGPUVirtualAddress());
	pCommandList->SetComputeRootDescriptorTable(ComputeRootSRVTable, srvHandle);
	pCommandList->SetComputeRootDescriptorTable(ComputeRootUAVTable, uavHandle);

	pCommandList->Dispatch(static_cast<int>(ceil(ParticleCount / 128.0f)), 1, 1);

	pCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pUavResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
}

void DXAsset::OnDestroy()
{
	// Notify the compute threads that the app is shutting down.
	InterlockedExchange(&m_terminating, 1);
	WaitForMultipleObjects(ThreadCount, m_threadHandles, TRUE, INFINITE);

	// Ensure that the GPU is no longer referencing resources that are about to be
	// cleaned up by the destructor.
	WaitForRenderContext();

	// Close handles to fence events and threads.
	CloseHandle(m_renderContextFenceEvent);
	for (int n = 0; n < ThreadCount; n++)
	{
		CloseHandle(m_threadHandles[n]);
		CloseHandle(m_threadFenceEvents[n]);
	}
}

void DXAsset::WaitForRenderContext()
{
	// Add a signal command to the queue.
	ThrowIfFailed(m_commandQueue->Signal(m_renderContextFence.Get(), m_renderContextFenceValue));

	// Instruct the fence to set the event object when the signal command completes.
	ThrowIfFailed(m_renderContextFence->SetEventOnCompletion(m_renderContextFenceValue, m_renderContextFenceEvent));
	m_renderContextFenceValue++;

	// Wait until the signal command has been processed.
	WaitForSingleObject(m_renderContextFenceEvent, INFINITE);
}

// Cycle through the frame resources. This method blocks execution if the 
// next frame resource in the queue has not yet had its previous contents 
// processed by the GPU.
void DXAsset::MoveToNextFrame()
{
	// Assign the current fence value to the current frame.
	m_frameFenceValues[m_frameIndex] = m_renderContextFenceValue;

	// Signal and increment the fence value.
	ThrowIfFailed(m_commandQueue->Signal(m_renderContextFence.Get(), m_renderContextFenceValue));
	m_renderContextFenceValue++;

	// Update the frame index.
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

	// If the next frame is not ready to be rendered yet, wait until it is ready.
	if (m_renderContextFence->GetCompletedValue() < m_frameFenceValues[m_frameIndex])
	{
		ThrowIfFailed(m_renderContextFence->SetEventOnCompletion(m_frameFenceValues[m_frameIndex], m_renderContextFenceEvent));
		WaitForSingleObject(m_renderContextFenceEvent, INFINITE);
	}
}


// Helper function for resolving the full path of assets.
std::wstring DXAsset::GetAssetFullPath(LPCWSTR assetName)
{
	return m_assetsPath + assetName;
}

// Helper function for acquiring the first available hardware adapter that supports Direct3D 12.
// If no such adapter can be found, *ppAdapter will be set to nullptr.
_Use_decl_annotations_
void DXAsset::GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter)
{
	ComPtr<IDXGIAdapter1> adapter;
	*ppAdapter = nullptr;

	for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter); ++adapterIndex)
	{
		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
		{
			// Don't select the Basic Render Driver adapter.
			// If you want a software adapter, pass in "/warp" on the command line.
			continue;
		}

		// Check to see if the adapter supports Direct3D 12, but don't create the
		// actual device yet.
		if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
		{
			break;
		}
	}

	*ppAdapter = adapter.Detach();
}

// Helper function for setting the window's title text.
void DXAsset::SetCustomWindowText(LPCWSTR text)
{
	std::wstring windowText = m_title + L": " + text;
	//SetWindowText(Window::GetHwnd(), windowText.c_str());
}

// Helper function for parsing any supplied command line args.
_Use_decl_annotations_
void DXAsset::ParseCommandLineArgs(WCHAR* argv[], int argc)
{
	for (int i = 1; i < argc; ++i)
	{
		if (_wcsnicmp(argv[i], L"-warp", wcslen(argv[i])) == 0 ||
			_wcsnicmp(argv[i], L"/warp", wcslen(argv[i])) == 0)
		{
			m_useWarpDevice = true;
			m_title = m_title + L" (WARP)";
		}
	}
}


// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Application
long long g_tick0, g_freq;
void zeroShitTime() {
	QueryPerformanceCounter((LARGE_INTEGER*)&g_tick0);
	QueryPerformanceFrequency((LARGE_INTEGER*)&g_freq);
}
double getShitTime() {
	long long g_tick1;
	QueryPerformanceCounter((LARGE_INTEGER*)&g_tick1);
	return double(g_tick1 - g_tick0) / double(g_freq);
}
//#define PTIME() printf("%.2lf: %s %d\n",getShitTime()*1000.0,__FILE__,__LINE__)
#define PTIME() 

void Application::run() {

	SDL_Event e;
	auto handle = [&]() {
		if (!_wm.eventHandle(e) && e.type == SDL_QUIT) {
			_quit = true;
			return false;
		}
		return true;
	};

	while (!_quit) {
		zeroShitTime();
		if (_idle) {
			while (SDL_PollEvent(&e) != 0) {
				if (!handle()) { break; }
			}

			PTIME();
			//if (SDL_PollEvent(&e) != 0) {
			//	PTIME();
			//	if (!handle()) { break; }
			//	PTIME();
			//}
			PTIME();
			_wm.idleEvent();
			PTIME();
			_wm.repaintEvent();
			PTIME();
		}
		else {
			int flag = SDL_WaitEvent(&e);
			if (flag) {
				handle();
			}
			else {
				printf("Application::run: SDL_WaitEvent error.\n");
			}
			while (SDL_PollEvent(&e) != 0) {
				if (!handle()) { break; }
			}			
			_wm.repaintEvent();
		}		
		
	}
}

void Application::init(Uint32 flags, int samples) {
	_init = !(SDL_Init(flags) < 0);
	if (!_init) {
		printf("SDLApp: %s\n", SDL_GetError());
	}

	auto ret = 0;

	// TODO:
	// move to window manager ?
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);

	auto flag = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, samples);
	if (flag) { 
		char emsg[128];
		sprintf(emsg, "SDL_GL_SetAttribute: cannot set SDL_GL_MULTISAMPLESAMPLES to %d\n", samples);
		throw std::runtime_error(emsg);
	}

	//SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	/* Turn on double buffering with a 24bit Z buffer.
	* You may need to change this to 16 or 32 for your system */	
	ret = SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	ret = SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	// ??? ogl init failed after adding this line. ???
	//ret = SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalLayout::HorizontalLayout(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent) {
	_sizePolicy = SizePolicy2(SP_AUTO, SP_AUTO);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

VerticalLayout::VerticalLayout(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent)
{
	_sizePolicy = SizePolicy2(SP_AUTO, SP_AUTO);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalSpacer::HorizontalSpacer(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent) 
{
	_sizePolicy = SizePolicy2(SP_EXPAND, SP_FIX);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

VerticalSpacer::VerticalSpacer(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent)
{
	_sizePolicy = SizePolicy2(SP_FIX, SP_EXPAND);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalScrollBar::HorizontalScrollBar(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{}

// -------- -------- -------- -------- -------- -------- -------- --------
VerticalScrollBar::VerticalScrollBar(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent)
{}

// -------- -------- -------- -------- -------- -------- -------- --------
ScrollWindow::ScrollWindow(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{}

// -------- -------- -------- -------- -------- -------- -------- --------
Frame::Frame(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{}

void Frame::paintEvent() {
	Painter painter(this);
	painter.strokeWidth((float)_borderWidth);
	painter.strokeColor(_borderColor);
	painter.fillColor(_backgroundColor);
	painter.rectangle(0.f, 0.f, (float)_size.x, (float)_size.y);
	paintChildren();
}

void Frame::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);	
	using PRIVATE::dget;
	_borderWidth = dget<int>(j, "border-width", 1);
	_border.left = _border.right = _border.top = _border.bottom = _borderWidth;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// UIHorizontalLine
HorizontalLine::HorizontalLine(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{
	_sizePolicy.y = SP_FIX;
	_size.y = _width;
}
void HorizontalLine::paintEvent() {
	Painter painter(this);
	painter.strokeWidth((float)_width);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	float y = _width * 0.5f;
	painter.line(0.f, y, (float)_size.x, y);
}
void HorizontalLine::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_width = dget<int>(j, "line-width", _width);
	_size.y = _width;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// UIVerticalLine
VerticalLine::VerticalLine(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{
	_sizePolicy.x = SP_FIX;
	_size.x = _width;
}
void VerticalLine::paintEvent() {
	Painter painter(this);
	painter.strokeWidth((float)_width);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	float x = _width * 0.5f;
	painter.line(x, 0.f, x, (float)_size.y);
}
void VerticalLine::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_width = dget<int>(j, "line-width", _width);
	_size.x = _width;
}

// -------- -------- -------- -------- -------- -------- -------- --------
Label::Label(const std::weak_ptr<SubWindow> &pparent) :SubWindow(pparent)
{
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void Label::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
}

void Label::paintEvent() {
	Painter painter(this);

	//painter.strokeColor(RGBA(255, 255, 255, 255));
	//painter.fillColor(RGBA(63, 63, 63, 255));
	//painter.rectangle(0, 0, _size.x, _size.y);

	IRect rect;
	rect.x = _border.left + _padding.left;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

IVec2 Label::minSize() {
	return IVec2(1, 20);
}

// -------- -------- -------- -------- -------- -------- -------- --------
PushButton::PushButton(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent){
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void PushButton::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
}

void PushButton::paintEvent() {

	Painter painter(this);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	painter.strokeWidth(1);

	if (_mouseIn) {
		painter.fillColor(RGBA(127, 127, 127, 255));
	}
	else {
		painter.fillColor(RGBA(63, 63, 63, 255));
	}
	painter.rectangle(0.f, 0.f, (float)_size.x, (float)_size.y);

	//painter.rectangle(1, 1, _size.x - 2, _size.y - 2);

	IRect rect;
	rect.x = _border.left + _padding.left;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

void PushButton::mouseLeftButtonDownEvent(int x, int y) {
	if (_onClick) { _onClick(); }
}

void PushButton::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void PushButton::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

IVec2 PushButton::minSize() {
	auto ms = _border.size2() + _padding.size2();
	Painter painter(this);
	auto tsize = painter.textSize(_text);
	//return IVec2(ms.w + tsize.x, ms.h + tsize.y);
	return IVec2(ms.w + tsize.x, 20);
}

void PushButton::onClick(std::function<void(void)> cb) {
	_onClick = cb;
}

// -------- -------- -------- -------- -------- -------- -------- --------
RadioButton::RadioButton(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent) {
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void RadioButton::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
	_value = dget<std::string>(j, "value") == "true" ? true : false;
}

void RadioButton::paintEvent() {

	Painter painter(this);

	//
	float cx = 10;
	float cy = 10;

	if (_mouseIn) {
		painter.strokeColor(0);
		painter.fillColor(RGBA(127, 127, 127, 255));
		painter.circle(cx, cy, 6.f);
	}

	painter.strokeColor(RGBA(255, 255, 255, 255));
	painter.fillColor(0);
	painter.circle(cx, cy, 6.f);

	if (_value) {
		painter.strokeColor(0);
		painter.fillColor(RGBA(255, 255, 255, 255));
		painter.circle(cx, cy, 3.f);
	}

	int width = 10 * (int)_text.size();

	IRect rect;
	rect.x = _border.left + _padding.left + 16;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

void RadioButton::mouseLeftButtonDownEvent(int x, int y) {
	_value = !_value;
	if (_onClick) { _onClick(_value); }
}

void RadioButton::onClick(std::function<void(bool)> cb) {
	_onClick = cb;
}

void RadioButton::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void RadioButton::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

// -------- -------- -------- -------- -------- -------- -------- --------
CheckBox::CheckBox(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent) {
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void CheckBox::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
	_value = dget<std::string>(j, "value") == "true" ? true : false;
}

void CheckBox::paintEvent() {
	Painter painter(this);
	
	float cx = 10;
	float cy = 10;
	float S = 6.f;

	painter.strokeColor(RGBA(255, 255, 255, 255));
	if (_mouseIn) {
		painter.fillColor(RGBA(127, 127, 127, 255));
	}
	else {
		painter.fillColor(RGBA(63, 63, 63, 255));
	}
	painter.rectangle(cx - S, cy - S, S * 2, S * 2);

	S = 3.f;
	if (_value) {
		painter.fillColor(RGBA(255, 255, 255, 255));
		painter.rectangle(cx - S, cy - S, S * 2, S * 2);
	}

	int width = 10 * (int)_text.size();

	IRect rect;
	rect.x = _border.left + _padding.left + 16;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

void CheckBox::mouseLeftButtonDownEvent(int x, int y) {
	_value = !_value;
	if (_onClick) { _onClick(_value); }
}

void CheckBox::onClick(std::function<void(bool)> cb) {
	_onClick = cb;
}

void CheckBox::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void CheckBox::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalSlider::HorizontalSlider(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent) {
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void HorizontalSlider::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_value = dget<int>(j, "value", _value);
	_maxValue = dget<int>(j, "max-value", _maxValue);
}

void HorizontalSlider::paintEvent() {
	Painter painter(this);
	if (_mouseIn) {
		painter.strokeWidth(5);
		painter.strokeColor(RGBA(127, 127, 127, 255));
		painter.line(10.f, 10.f, _size.x - 10.f, 10.f);
	}
	painter.strokeWidth(1);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	painter.line(10.f, 10.f, _size.x - 10.f, 10.f);
	painter.strokeColor(RGBA(255, 255, 255, 0));
	painter.fillColor(RGBA(255, 255, 255, 255));
	painter.circle(10 + (_size.x - 20) * (float)_value / (float)_maxValue, 10, 6);
}

void HorizontalSlider::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void HorizontalSlider::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Painter Base

// -------- -------- -------- -------- -------- -------- -------- --------
PainterBase::PainterBase(SubWindow *psw) {
	if (!psw) { return; }
	init(psw);
}

// -------- -------- -------- -------- -------- -------- -------- --------
PainterBase::PainterBase(std::weak_ptr<SubWindow> &wpsw) {
	auto psw = wpsw.lock();
	if (!psw) { return; }
	init(psw.get());
}

// -------- -------- -------- -------- -------- -------- -------- --------
PainterBase::~PainterBase() {
	update();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void PainterBase::init(SubWindow *psw) {

	GLGUI_CHECK_GL_ERROR();

	// Get paint data.
	_data = psw->paintData().lock();
	_sharedData = psw->rootWindow().lock()->sharedPaintData();

	// Set viewport.
	const IVec2 &pos = psw->position();
	const IVec2 &size = psw->size();
	_size = size;
	IVec2 rootSize;
	auto prw = psw->rootWindow().lock();
	if (!prw) {
		printf("PainterBase::init: SubWindow doesn't have a root.\n");
	}
	rootSize = prw ? prw->size() : size;

	int x = pos.x;
	int y = rootSize.y - (pos.y + size.y);
	int w = std::max(size.x, 0);
	int h = std::max(size.y, 0);

	//
	glEnable(GL_SCISSOR_TEST);
	glDisable(GL_DEPTH_TEST);
	glScissor(x, y, w, h);
	glViewport(x, y, w, h);
	if (w == 0 || h == 0) { return; }

	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
	//glOrtho(0, w, 0, h, -1, 1);

	//glTranslatef(0, h, 0);
	//glScalef(0, -1, 0);

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void PainterBase::update() {
	if (!_update) { return; }
	_update = false;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void PainterBase::strokeColor(const RGBA &c) {
	_strokeColor = c;
}
void PainterBase::strokeColor(uint8_t a) {
	_strokeColor.a = a;
}
void PainterBase::strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_strokeColor = RGBA(r,g,b,a);
}

void PainterBase::strokeWidth(float w) {
	_strokeWidth = w;
}

void PainterBase::fillColor(const RGBA &c) {
	_fillColor = c;
}
void PainterBase::fillColor(uint8_t a) {
	_fillColor.a = a;
}
void PainterBase::fillColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_fillColor = RGBA(r, g, b, a);
}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// GLES painter.
#ifdef GLGUI_USE_GLES_PAINTER

void GLESSharedPaintData::init() {

	// font
	if (!fontBuffer) {
#ifdef ENABLE_STB_TEXT
		auto &ttf_buffer = fontBuffer;
		if (!ttf_buffer) { ttf_buffer = new uint8_t[50000]; }
		//auto &font = font;
		fread(ttf_buffer, 1, 50000, fopen("./font/DroidSans.ttf", "rb"));
		//fread(ttf_buffer, 1, 50000, fopen("./font/DroidSans-Bold.ttf", "rb"));
		stbtt_InitFont(&font, ttf_buffer, stbtt_GetFontOffsetForIndex(ttf_buffer, 0));
#endif
	}

	// shader
	std::string uiVertexShader =
		"#version 400 \n"
		"uniform float width; \n"
		"uniform float height; \n"
		"layout(location = 0) in vec2 i_vertex; \n"
		"layout(location = 1) in vec4 i_color; \n"
		"flat out vec4 fragColor; \n"		
		"void main(){ \n"
		"    fragColor = i_color; \n"
		"    gl_Position = vec4(i_vertex.x/width*2.0-1.0, i_vertex.y/height*2.0-1.0, 0.0, 1.0);"
		"}";

	std::string uiFragmentShader =
		"#version 400 \n"
		"layout(location = 0) out vec4 color; \n"
		"flat in vec4 fragColor; \n"
		"void main(){ \n"
		"    color = fragColor;"
		"}";

	std::string textVertexShader =
		"#version 400 \n"
		"uniform float width; \n"
		"uniform float height; \n"
		"layout(location = 0) in vec2 i_vertex; \n"
		"layout(location = 1) in vec2 i_texcoord; \n"
		"out vec2 texcoord; \n"		
		"void main(){ \n"
		"    texcoord = i_texcoord; \n"
		"    gl_Position = vec4(i_vertex.x/width*2.0-1.0, i_vertex.y/height*2.0-1.0, 0.0, 1.0);"
		"}";

	std::string textFragmentShader =
		"#version 400 \n"
		"uniform sampler2D ftex; \n"
		"in vec2 texcoord; \n"
		"layout(location = 0) out vec4 color; \n"
		"void main(){ \n"
		"    color = texture(ftex, texcoord);"
		"}";

	uiShader
		.createShader(GLPP::Vertex, uiVertexShader)
		.createShader(GLPP::Fragment, uiFragmentShader)
		.link();

	textShader
		.createShader(GLPP::Vertex, textVertexShader)
		.createShader(GLPP::Fragment, textFragmentShader)
		.link();

	_ready = true;
}

// -------- -------- -------- -------- -------- -------- -------- --------
GLESPainter::GLESPainter(SubWindow *psw)
	:PainterBase(psw) {

	auto &uiShader = _sharedData->uiShader;
	if (!uiShader.linked()) {uiShader.link(); }
	uiShader.uniform1f("width", (float)_size.x);
	uiShader.uniform1f("height", (float)_size.y);

	auto &textShader = _sharedData->textShader;
	if (!textShader.linked()) { textShader.link(); }
	textShader.uniform1f("width", (float)_size.x);
	textShader.uniform1f("height", (float)_size.y);
	textShader.uniform1i("ftex", 0);

	glUseProgram(0);

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
GLESPainter::GLESPainter(std::weak_ptr<SubWindow> &wpsw)
	:PainterBase(wpsw){
}

// -------- -------- -------- -------- -------- -------- -------- --------
GLESPainter::~GLESPainter() {
	update();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::update() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::line(float x0, float y0, float x1, float y1) {
	
	GLGUI_CHECK_GL_ERROR();

	if (_strokeColor.a) {

		float vd[4] = {
			x0, y0, x1, y1
		};
		float cd[8] = {
			_strokeColor.r / 255.f, _strokeColor.g / 255.f, _strokeColor.b / 255.f, _strokeColor.a / 255.f,
			_strokeColor.r / 255.f, _strokeColor.g / 255.f, _strokeColor.b / 255.f, _strokeColor.a / 255.f,
		};

		glLineWidth(_strokeWidth);

		_sharedData->uiShader.use();

		GLuint va;
		glGenVertexArrays(1, &va);
		glBindVertexArray(va);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		GLuint vb, cb;
		glGenBuffers(1, &vb);
		glGenBuffers(1, &cb);
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), vd, GL_STATIC_DRAW);
		
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), cd, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDrawArrays(GL_LINES, 0, 2);
		glFinish();

		glBindVertexArray(0);

		glDeleteBuffers(1, &vb);
		glDeleteBuffers(1, &cb);
		glDeleteVertexArrays(1, &va);

		glUseProgram(0);
	}

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::circle(float cx, float cy, float r) {

	_sharedData->uiShader.use();

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	GLuint vb, cb;
	glGenBuffers(1, &vb);
	glGenBuffers(1, &cb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, cb);
	glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	static const double PI = acos(-1.0);
	glLineWidth(1);

	if (_strokeColor.a) {
		std::vector<float> vd;
		std::vector<float> cd;

		for (int i = 0; i <= 64; ++i) {
			vd.push_back((float)(cx + sin(i / 32.0 *PI) * r));
			vd.push_back((float)(cy + cos(i / 32.0 *PI) * r));
			cd.push_back(_strokeColor.r/255.f);
			cd.push_back(_strokeColor.g / 255.f);
			cd.push_back(_strokeColor.b / 255.f);
			cd.push_back(_strokeColor.a / 255.f);
		}
		int32_t s = (int32_t)vd.size();
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), vd.data(), GL_STATIC_DRAW);
		s = (int32_t)cd.size();
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), cd.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDrawArrays(GL_LINE_STRIP, 0, s / 2);
		glFinish();
	}

	if (_fillColor.a) {
		std::vector<float> vd;
		std::vector<float> cd;
		vd.push_back(cx);
		vd.push_back(cy);
		cd.push_back(_fillColor.r/ 255.f);
		cd.push_back(_fillColor.g / 255.f);
		cd.push_back(_fillColor.b / 255.f);
		cd.push_back(_fillColor.a / 255.f);
		for (int i = 0; i <= 64; ++i) {
			vd.push_back((float)(cx + sin(i / 32.0 *PI) * r));
			vd.push_back((float)(cy + cos(i / 32.0 *PI) * r));
			cd.push_back(_fillColor.r / 255.f);
			cd.push_back(_fillColor.g / 255.f);
			cd.push_back(_fillColor.b / 255.f);
			cd.push_back(_fillColor.a / 255.f);
		}
		int32_t s = (int32_t)vd.size();
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), vd.data(), GL_STATIC_DRAW);
		s = (int32_t)cd.size();
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), cd.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDrawArrays(GL_TRIANGLE_FAN, 0, s / 2);
		glFinish();
	}

	glBindVertexArray(0);

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &cb);
	glDeleteVertexArrays(1, &va);

	glUseProgram(0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::rectangle(float x, float y, float w, float h) {

	_sharedData->uiShader.use();

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	GLuint vb, cb;
	glGenBuffers(1, &vb);
	glGenBuffers(1, &cb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, cb);
	glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (_strokeColor.a && (_strokeWidth>0)) {

		//glColor4ub(_strokeColor.r, _strokeColor.g, _strokeColor.b, _strokeColor.a);

		float vd[8] = {
			x, y,
			x + w, y,
			x + w, y + h,
			x, y + h
		};

		std::vector<float> cd;
		cd.push_back(_strokeColor.r / 255.f);
		cd.push_back(_strokeColor.g / 255.f);
		cd.push_back(_strokeColor.b / 255.f);
		cd.push_back(_strokeColor.a / 255.f);
		cd.insert(cd.end(), cd.begin(), cd.end());
		cd.insert(cd.end(), cd.begin(), cd.end());
		cd.insert(cd.end(), cd.begin(), cd.end());
		
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vd, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, 32 * sizeof(float), cd.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_QUADS, 0, 4);
		glFinish();

		auto &s = _strokeWidth;
		x += s;
		y += s;
		w -= s * 2;
		h -= s * 2;
	}

	float vd[8] = {
		x, y,
		x + w, y,
		x + w, y + h,
		x, y + h
	};

	std::vector<float> cd;
	cd.push_back(_fillColor.r / 255.f);
	cd.push_back(_fillColor.g / 255.f);
	cd.push_back(_fillColor.b / 255.f);
	cd.push_back(_fillColor.a / 255.f);
	cd.insert(cd.end(), cd.begin(), cd.end());
	cd.insert(cd.end(), cd.begin(), cd.end());
	cd.insert(cd.end(), cd.begin(), cd.end());

	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vd, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, cb);
	glBufferData(GL_ARRAY_BUFFER, 32 * sizeof(float), cd.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_QUADS, 0, 4);
	glFinish();

	//
	glBindVertexArray(0);

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &cb);
	glDeleteVertexArrays(1, &va);

	glUseProgram(0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
#ifdef ENABLE_STB_TEXT
stbtt_fontinfo &GLESPainter::getSharedFont() {
	auto ps = _sharedData;
	if (!ps->ready()) {
		ps->init();
	}
	return ps->font;
}

// -------- -------- -------- -------- -------- -------- -------- --------
IVec2 GLESPainter::textSize(const stbtt_fontinfo &font, const std::string &t) {
#ifdef ENABLE_STB_TEXT
	int ascent, descent, advance, lsb;
	float xoffset = 0.f, scale = stbtt_ScaleForPixelHeight(&font, 16);
	stbtt_GetFontVMetrics(&font, &ascent, &descent, 0);
	for (int ch = 0; t[ch]; ++ch) {
		float x_shift = xoffset - (float)floor(xoffset);
		stbtt_GetCodepointHMetrics(&font, t[ch], &advance, &lsb);
		stbtt_GetCodepointBitmapBoxSubpixel(&font, t[ch], scale, scale, x_shift, 0, 0, 0, 0, 0);
		xoffset += (advance * scale);
		if (t[ch + 1]) {
			xoffset += scale*stbtt_GetCodepointKernAdvance(&font, t[ch], t[ch + 1]);
		}
	}
	int SH = (int)((std::abs(ascent) + std::abs(descent)) * scale);
	int SW = (int)(xoffset + 1);
	return IVec2(SW, SH);
#else
	return IVec2(0, 0);
#endif
}

IVec2 GLESPainter::textSize(const std::string &t) {
	auto &font = getSharedFont();
	return textSize(font, t);
}

#else
IVec2 GLESPainter::textSize(const std::string &t) {
	return IVec2(0, 0);
}
#endif

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::text(float x, float y, const std::string &t) {
#ifdef ENABLE_STB_TEXT
	GLGUI_CHECK_GL_ERROR();

	// 
	auto &font = getSharedFont();
	auto scale = stbtt_ScaleForPixelHeight(&font, 16);
	int ascent, descent, lineGap;
	stbtt_GetFontVMetrics(&font, &ascent, &descent, &lineGap);
	auto baseline = (int)(ascent*scale);

	//
	float xoffset = 0.f;
	int ch = 0;
	for (int ch = 0; t[ch]; ++ch) {
		int advance, lsb;
		float x_shift = xoffset - (float)floor(xoffset);
		stbtt_GetCodepointHMetrics(&font, t[ch], &advance, &lsb);
		stbtt_GetCodepointBitmapBoxSubpixel(&font, t[ch], scale, scale, x_shift, 0, 0, 0, 0, 0);
		xoffset += (advance * scale);
		if (t[ch + 1]) {
			xoffset += scale*stbtt_GetCodepointKernAdvance(&font, t[ch], t[ch + 1]);
		}
	}
	int SH = (int)((std::abs(ascent) + std::abs(descent)) * scale);
	int SW = (int)(xoffset + 1);

	uint8_t *screen = new uint8_t[SW*SH]; 
	memset(screen, 0, SW*SH);

	float xpos = 0;
	ch = 0;
	while (t[ch]) {
		int advance, lsb, x0, y0, x1, y1;
		float x_shift = xpos - (float)floor(xpos);
		stbtt_GetCodepointHMetrics(&font, t[ch], &advance, &lsb);
		stbtt_GetCodepointBitmapBoxSubpixel(&font, t[ch], scale, scale, x_shift, 0, 
			&x0, &y0, &x1, &y1);
		stbtt_MakeCodepointBitmapSubpixel(&font, &screen[(baseline + y0) * SW + (int)xpos + x0], 
			x1 - x0, y1 - y0, SW, scale, scale, x_shift, 0, t[ch]);
		// note that this stomps the old data, so where character boxes overlap (e.g. 'lj') it's wrong
		// because this API is really for baking character bitmaps into textures. if you want to render
		// a sequence of characters, you really need to render each bitmap to a temp buffer, then
		// "alpha blend" that into the working buffer
		xpos += (advance * scale);
		if (t[ch + 1])
			xpos += scale*stbtt_GetCodepointKernAdvance(&font, t[ch], t[ch + 1]);
		++ch;
	}

	uint8_t *rgba = new uint8_t[SW*SH * 4];
	for (int j = 0; j < SH; ++j) {
		for (int i = 0; i < SW; ++i) {
			auto p = j*SW + i;
			auto p4 = p * 4;
			auto v = screen[p];
			if (v) {
				rgba[p4] = rgba[p4 + 1] = rgba[p4 + 2] = 255;
				rgba[p4 + 3] = v;
			}
			else {
				rgba[p4] = rgba[p4 + 1] = rgba[p4 + 2] = rgba[p4 + 3] = 0;
			}
		}
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_TEXTURE_2D);

	GLuint tex = 0;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SW, SH, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

	_sharedData->textShader.use();

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	GLuint vb, tb;
	glGenBuffers(1, &vb);
	glGenBuffers(1, &tb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, tb);
	glVertexAttribPointer(1, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	float vd[8] = {
		x, y + SH,
		x + SW, y + SH,
		x + SW, y,
		x, y
	};

	float td[8] = {
		0.f, 0.f, 
		1.f, 0.f, 
		1.f, 1.f,
		0.f, 1.f
	};



	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vd, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, tb);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), td, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_QUADS, 0, 4);
	glFinish();

	//glBegin(GL_QUADS);
	//glTexCoord2f(0.f, 0.f);
	//glVertex2f(x, y+SH);
	//
	//glTexCoord2f(1.f, 0.f);
	//glVertex2f(x+SW, y+SH);
	//
	//glTexCoord2f(1.f, 1.f);
	//glVertex2f(x+SW, y);

	//glTexCoord2f(0.f, 1.f);
	//glVertex2f(x, y);

	//glEnd();
	//glFinish();

	glBindTexture(GL_TEXTURE_2D, 0);

	glDeleteTextures(1, &tex);

	delete[] screen;
	delete[] rgba;

	//
	glBindVertexArray(0);

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &tb);
	glDeleteVertexArrays(1, &va);

	glUseProgram(0);

	GLGUI_CHECK_GL_ERROR();
#endif
}

void GLESPainter::text(const IRect &rect, const std::string &t) {
	text(rect, t, HA_CENTER, VA_CENTER);
}
void GLESPainter::text(const IRect &rect, const std::string &t, VerticalAlign va) {
	text(rect, t, HA_CENTER, va);
}
void GLESPainter::text(const IRect &rect, const std::string &t, HorizontalAlign ha) {
	text(rect, t, ha, VA_CENTER);
}

void GLESPainter::text(const IRect &rect, const std::string &t,
	HorizontalAlign ha, VerticalAlign va) {

	static const int W = 10;
	static const int H = 16;

	// default font: w 10 h 16
	float x, y;

	if (ha == HA_LEFT) {
		x = (float)rect.x;
	}
	else if (ha == HA_RIGHT) {
		x = (float)(rect.x + (rect.width - (int)t.size() * W));
	}
	else { // center
		x = rect.x + (rect.width - (int)t.size() * W) / 2.f;
	}

	if (va == VA_TOP) {
		y = (float)(rect.y + (rect.height - H));
	}
	else if (va == VA_BOTTOM) {
		y = (float)(rect.y + rect.height);
	}
	else { // center
		y = rect.y + (rect.height - H) / 2.f;
	}

	text(x, y, t);
}

#endif

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// OpenGL painter.
#ifdef GLGUI_USE_GL_PAINTER
#endif

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// NVIDIA_path_rendering painter.
#ifdef GLGUI_USE_NVPR_PAINTER

// -------- -------- -------- -------- -------- -------- -------- --------
NVPRPainter::NVPRPainter(SubWindow *psw) {
	if (!psw) { return; }
	_data = psw->paintData().lock();
}

// -------- -------- -------- -------- -------- -------- -------- --------
NVPRPainter::NVPRPainter(std::weak_ptr<SubWindow> &wpsw) {
	auto psw = wpsw.lock();
	if (!psw) { return; }
	_data = psw->paintData().lock();
}

// -------- -------- -------- -------- -------- -------- -------- --------
NVPRPainter::~NVPRPainter() {
	update();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::clear() {
	//glDeletePathsNV()
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::update() {
	if (!_update) { return; }
	_update = false;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::strokeColor(uint8_t a){
	_sa = a;
}
void NVPRPainter::fillColor(uint8_t a){
	_fa = a;
}
void NVPRPainter::strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_sr = r; _sg = g; _sb = b; _sa = a;
}
void NVPRPainter::fillColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_fr = r; _fg = g; _fb = b; _fa = a;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::addPath(GLsizei ncmd, GLubyte *cmd, GLsizei ncoord, GLfloat *coord) {
	//GLuint path = glGenPathsNV(1);
	//if (!path) { 
	//	return;
	//}
	//glPathCommandsNV(path, ncmd, cmd, ncoord, GL_FLOAT, coord);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::line() {
	//static const GLubyte pathCommands[10] =
	//{ GL_MOVE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV,
	//GL_LINE_TO_NV, GL_CLOSE_PATH_NV,
	//'M', 'C', 'C', 'Z' };  // character aliases
	//static const GLshort pathCoords[12][2] =
	//{ { 100, 180 }, { 40, 10 }, { 190, 120 }, { 10, 120 }, { 160, 10 },
	//{ 300, 300 }, { 100, 400 }, { 100, 200 }, { 300, 100 },
	//{ 500, 200 }, { 500, 400 }, { 300, 300 } };
	//glPathCommandsNV(pathObj, 10, pathCommands, 24, GL_SHORT, pathCoords);

	/* Before rendering, configure the path object with desirable path
	parameters for stroking.  Specify a wider 6.5-unit stroke and
	the round join style: */

	//glPathParameteriNV(pathObj, GL_PATH_JOIN_STYLE_NV, GL_ROUND_NV);
	//glPathParameterfNV(pathObj, GL_PATH_STROKE_WIDTH_NV, 6.5);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::rectangle(int x, int y, int w, int h) {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::paint(std::shared_ptr<NVPRPaintData> &pd) {

	if (!pd) { return; }
	auto &data = *pd;

	int pathNumber = (int)data.glPath.size();
	for (int i = 0; i < pathNumber; ++i) {

		//glClearStencil(0);
		//glStencilMask(~0);
		//glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		//glMatrixLoadIdentityEXT(GL_PROJECTION);
		//glMatrixLoadIdentityEXT(GL_MODELVIEW);
		//glMatrixOrthoEXT(GL_MODELVIEW, 0, 500, 0, 400, -1, 1);

		GLuint pathObj = data.glPath[i];
		auto fillColor = data.pathFillColor[i];
		auto strokeColor = data.pathStrokeColor[i];

		bool filling = !!(fillColor >> 24);
		bool stroking = !!(strokeColor >> 24);

		if (filling) {

			/* Stencil the path: */

			//glStencilFillPathNV(pathObj, GL_COUNT_UP_NV, 0x1F);

			/* The 0x1F mask means the counting uses modulo-32 arithmetic. In
			principle the star's path is simple enough (having a maximum winding
			number of 2) that modulo-4 arithmetic would be sufficient so the mask
			could be 0x3.  Or a mask of all 1's (~0) could be used to count with
			all available stencil bits.

			Now that the coverage of the star and the heart have been rasterized
			into the stencil buffer, cover the path with a non-zero fill style
			(indicated by the GL_NOTEQUAL stencil function with a zero reference
			value): */

			//glEnable(GL_STENCIL_TEST);
			//glStencilFunc(GL_NOTEQUAL, 0, 0x1F);

			//glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
			//glColor3f(0, 1, 0); // green
			//glCoverFillPathNV(pathObj, GL_BOUNDING_BOX_NV);

		}

		/* The result is a yellow star (with a filled center) to the left of
		a yellow heart.

		The GL_ZERO stencil operation ensures that any covered samples
		(meaning those with non-zero stencil values) are zero'ed when
		the path cover is rasterized. This allows subsequent paths to be
		rendered without clearing the stencil buffer again.

		A similar two-step rendering process can draw a white outline
		over the star and heart. */

		/* Now stencil the path's stroked coverage into the stencil buffer,
		setting the stencil to 0x1 for all stencil samples within the
		transformed path. */

		if (stroking) {

			//glStencilStrokePathNV(pathObj, 0x1, ~0);

			/* Cover the path's stroked coverage (with a hull this time instead
			of a bounding box; the choice doesn't really matter here) while
			stencil testing that writes white to the color buffer and again
			zero the stencil buffer. */

			//glColor3f(1, 1, 0); // yellow
			//glCoverStrokePathNV(pathObj, GL_CONVEX_HULL_NV);

			/* In this example, constant color shading is used but the application
			can specify their own arbitrary shading and/or blending operations,
			whether with Cg compiled to fragment program assembly, GLSL, or
			fixed-function fragment processing.

			More complex path rendering is possible such as clipping one path to
			another arbitrary path.  This is because stencil testing (as well
			as depth testing, depth bound test, clip planes, and scissoring)
			can restrict path stenciling. */
		}
	}

	int textNumber = (int)data.text.size();
	for (int i = 0; i < textNumber; ++i) {

		/* STEP 1: stencil message into stencil buffer.  Results in samples
		within the message's glyphs to have a non-zero stencil value. */

		//glDisable(GL_STENCIL_TEST);
		//glStencilFillPathInstancedNV(numUTF8chars,
		//	GL_UTF8_NV, koreanName, glyphBase,
		//	GL_PATH_FILL_MODE_NV, ~0,  // Use all stencil bits
		//	GL_TRANSLATE_X_NV, xoffset);

		/* STEP 2: cover region of the message; color covered samples (those
		with a non-zero stencil value) and set their stencil back to zero. */

		//glEnable(GL_STENCIL_TEST);
		//glStencilFunc(GL_NOTEQUAL, 0, ~0);
		//glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

		//glColor3ub(192, 192, 192);  // gray
		//glCoverFillPathInstancedNV(numUTF8chars,
		//	GL_UTF8_NV, koreanName, glyphBase,
		//	GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
		//	GL_TRANSLATE_X_NV, xoffset);
	}

}

#endif

} // end of namespace GLGUI
} // end of namespace Mochimazui
