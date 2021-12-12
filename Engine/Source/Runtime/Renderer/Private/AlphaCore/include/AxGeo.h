#ifndef __AX_GEO_H__
#define __AX_GEO_H__

#include <Utility/AxStorage.h>

struct AxTopolgy
{
	AxBufferUInt32 Indices;
	AxBuffer2I Primitives;
	AxBufferV3 Position;
};

class AxGeometry
{
public:
	AxGeometry();
	~AxGeometry();

	AxTopolgy& GetTopolgy() { return m_Topolgy; };

	void Save(std::string path);
private:
	
	AxTopolgy m_Topolgy;


};

#endif