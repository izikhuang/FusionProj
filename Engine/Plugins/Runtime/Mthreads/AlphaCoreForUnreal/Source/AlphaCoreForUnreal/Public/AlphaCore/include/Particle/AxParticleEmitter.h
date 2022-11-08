#ifndef __AX_PARTICLE_EMITTER_H__
#define __AX_PARTICLE_EMITTER_H__

#include "Utility/AxStorage.h"
#include "AxGeo.h"

class AxParticleEmitter
{
public:
	struct RAWDesc
	{

	};
	AxParticleEmitter();
	~AxParticleEmitter();

	void Birth(AxUInt32 birthNum);
private:

	AxGeometry* m_OwnGeometry;
	
};


#endif