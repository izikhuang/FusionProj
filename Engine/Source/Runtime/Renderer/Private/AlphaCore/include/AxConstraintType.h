#ifndef __AX_CONSTRAINT_TYPE_H__
#define __AX_CONSTRAINT_TYPE_H__

namespace AlphaCore
{
	namespace SolidUtility
	{
		enum AxSolidConstraint
		{
			kDistance				= 1,
			kBend					= 2,
			kAttach					= 3,
			kTetrahedraVolume		= 4,
			kStretchShear			= 5,
			kBendTwist				= 6,
			kTriangleStretch		= 7,
			kInvalidSolidConstraint = 0
		};
	}
}



#endif
