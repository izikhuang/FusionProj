/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2018 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "../../SDL_internal.h"

extern int SDL_DINPUT_JoystickInit(void);
extern void SDL_DINPUT_JoystickDetect(JoyStick_DeviceData **pContext);
extern int SDL_DINPUT_JoystickOpen(SDL_Joystick * joystick, JoyStick_DeviceData *joystickdevice);
extern int SDL_DINPUT_JoystickRumble(SDL_Joystick * joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble, Uint32 duration_ms);
extern void SDL_DINPUT_JoystickUpdate(SDL_Joystick * joystick);
extern void SDL_DINPUT_JoystickClose(SDL_Joystick * joystick);
extern void SDL_DINPUT_JoystickQuit(void);

/* vi: set ts=4 sw=4 expandtab: */
