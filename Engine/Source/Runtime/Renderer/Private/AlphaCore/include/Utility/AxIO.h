#ifndef __ALPHA_CORE_UTILITY_IO_H__
#define __ALPHA_CORE_UTILITY_IO_H__

#include <string>
#include <Math/AxVectorBase.h>

namespace AlphaUtility
{
	int NumFrameSize(std::string frameRaw);
	std::string EvalFileName(std::string filename, int frame);
	void ReplaceString(std::string &strBase, std::string strSrc, std::string strDes);

	void SaveVector3(std::ifstream& ifs,AxVector3 v3);
	void ReadVector3(std::ofstream& ofs,AxVector3& v3);
}

#endif