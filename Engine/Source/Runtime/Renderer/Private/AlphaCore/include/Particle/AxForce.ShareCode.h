#ifndef __AX_FORCE_SHARECODE_H__
#define __AX_FORCE_SHARECODE_H__

#include <AxDataType.h>
#include <AxMacro.h>
#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <ProceduralContent/AxNoise.ShareCode.h>
#include <Math/AxMat.h>
#include <Math/AxMat.ShareCode.h>
#include <Math/AxMath101.h>
namespace AlphaCore 
{
	namespace Particle 
	{
		namespace ShareCode 
		{
			ALPHA_SHARE_FUNC void ImplicitVelocityDamping(
				AxUInt32 idx,
				AxVector3* posRaw,
				AxVector3* velRaw,
				AxVector2UI* p2pMap2IRaw,
				AxUInt32* p2pMapIndicesRaw,
				AxVector3* targetVelocityRaw,
				AxFp32* airResistRaw,
				AxFp32* normalDragRaw,
				AxFp32* targetDragRaw,
				AxFp32* massRaw,
				AxFp32 dt)
			{
				AxVector3 v = velRaw[idx];
				AxVector3 targetv = targetVelocityRaw[idx];
				AxFp32 airresist = airResistRaw[idx];
				AxFp32 dragnormal = normalDragRaw[idx];
				AxFp32 dragtangent = targetDragRaw[idx];
				// Get the relative ang vel
				v -= targetv;
				///airresist *= invmass;
				AxFp32 invMass = (massRaw == nullptr) ? 1.0f : (1.0f / massRaw[idx]);
				airresist *= invMass;
				if (airresist == 0)
					return;
				//printf("air:%f", airresist);

				AxFp32 dragexp = 2;
				AxVector3 dragShape = MakeVector3(1.0f, 1.0f, 1.0f);
				Quat qrot;
				bool hasdragshape = false;
				if (hasdragshape)
				{
					//lcl_orient = p@orient;
					//printf("hasdragshape------hasdragshape");
				}
				else if (dragnormal != 1.0f || dragtangent != 1.0f)
				{
					// Enable drag shape
					hasdragshape = true;
					//printf("Enable drag shape ! \n");
					// Compute our local orient from neighbourhood.
					AxVector3 pos = posRaw[idx];
					AxVector3 vel = velRaw[idx];
					AxVector2UI startNum = p2pMap2IRaw[idx];
					AxMat3x3F g = MakeMat3x3F();
					AxFp32 dragNormal = normalDragRaw[idx];
					AxFp32 dragTangent = targetDragRaw[idx];
					AX_FOR_K(startNum.y)
					{
						int linkedId = p2pMapIndicesRaw[startNum.x + k];
						AxVector3 linkedPos = posRaw[linkedId];
						linkedPos -= pos;
						linkedPos = Normalize(linkedPos);
						g += AlphaCore::Math::ShareCode::Outerproduct(linkedPos, linkedPos);
					}
 					g /= (AxFp32)(startNum.y);
					//PrintInfo("G:", g,true);

					AxVector3 diag;
					AxMat3x3F rot;
					AlphaCore::Math::ShareCode::DiagonalizeSymmetricMatrix(g, rot, diag);
					//PrintInfo("rot:", rot, true);

					qrot = MakeQuatFormMat3x3(rot);
					//PrintInfo("diag:", diag);

					dragShape = AlphaCore::Math::Fit(diag, 0, 0.5f, dragNormal, dragTangent);
					//PrintInfo("dragShape:", dragShape);

				}

				if (hasdragshape)
				{
					Quat back = qrot;
					back = QuatInverse(back);
					//
					//v = QuatRotate(back, v); // TODO Diff???
					//
					v = back * v;
					if (dragexp == 2)
					{
						// Only quadratic
						AxVector3 scale = airresist * dt * abs(v) * dragShape;
						//PrintInfo("abs(v)", abs(v));
						scale += 1.0f;
						scale = 1.0f / scale;
						v *= scale;
					}
					else if (dragexp == 1)
					{
						// Only linear
						//AxVector3 scale = dt * airresist * dragshape;
						//scale = exp(-scale);
						//v *= scale;
					}
					else
					{
						// Both!
						//vector a = 2 - dragexp;
						//vector b = dragexp - 1;
						//vector v0 = abs(v);
						//
						//a *= airresist * dragshape;
						//b *= airresist * dragshape;
						//
						//vector t0 = log(a / (b * v0) + 1) / a;
						//vector vnew = a / (b * (exp(a*(t0 + tinc)) - 1));
						//vector scale = vnew / v0;
						//v *= scale;
					}
					// vel = qrot * vel;

					/// TODO v = QuatRotate(qrot, v);
					v = qrot * v;
				}
				else
				{
					if (dragexp == 2)
					{
						// Only Quadratic
						AxFp32 scale = airresist * dt * Length(v);
						scale += 1;
						scale = 1 / scale;
						v *= scale;
					}
					else if (dragexp == 1)
					{
						// Only Linear
						float scale = dt * airresist;
						scale = exp(-scale);
						v *= scale;
					}
					else
					{
						// Both!
						AxFp32 a = 2 - dragexp;
						AxFp32 b = dragexp - 1;
						AxFp32 v0 = Length(v);

						a *= airresist;
						b *= airresist;

						AxFp32 t0 = log(a / (b * v0) + 1) / a;
						AxFp32 vnew = a / (b * (exp(a*(t0 + dt)) - 1));
						AxFp32 scale = vnew / v0;
						v *= scale;
					}
				}
				// Restore frame
				v += targetv;
				velRaw[idx] = v;
			}

			namespace Internal 
			{
				ALPHA_SHARE_FUNC void AddWindForce(
					AxVector3 addWindVelocity,
					AxFp32 addAirResist,
					AxVector3& oldWindVelocityRaw,
					AxFp32& oldAirResistRaw)
				{
					AxVector3 addWindVel = addWindVelocity * addAirResist;
					AxVector3 oldWindVel = oldWindVelocityRaw * oldAirResistRaw;
					AxFp32 newAirResist = oldAirResistRaw + addAirResist;
					AxVector3 retWindVel = (newAirResist == 0.0f) ? MakeVector3() : (addWindVel + oldWindVel) / newAirResist;
					oldWindVelocityRaw = retWindVel;
					oldAirResistRaw = newAirResist;
				}
			}

			ALPHA_SHARE_FUNC void WindForce(AxUInt32 idx,
				AxCurlNoiseParam curlNoiseParam,
				AxVector3* posRaw,
				AxVector3* windVelRaw,
				AxFp32* windAirresistRaw,
				AxFp32* massRaw,
				AxFp32* windIntensityBuf,
				AxVector3 windVelocityParam,
				AxFp32 windAirresistParam,
				bool ignoreMass, // ignoreMass = Mass as a scale factor 
				AxFp32 time,
				AxFp32 dt)
			{
				AxVector3 pos	  = posRaw[idx];
				AxVector3 windVel = windVelRaw[idx];
				AxFp32 windResist = windAirresistRaw[idx];
				AxFp32* controlPropRaw = curlNoiseParam.ControlProperty;
				AxFp32 controlVal = controlPropRaw == nullptr ? 0.0f : controlPropRaw[idx];
				AxVector3 noise = AlphaCore::ProceduralContent::ShareCode::Internal::CurlNoise4DVector(
					curlNoiseParam,
					pos,
					controlVal,
					curlNoiseParam.NoiseData,time,dt);

				AxFp32 resistScale = ignoreMass ? massRaw[idx] : 1.0f;
				AxFp32 finalWindScale = windIntensityBuf == nullptr ? 1.0f : windIntensityBuf[idx];
				windAirresistParam *= resistScale * finalWindScale;
				Internal::AddWindForce(
					(windVelocityParam + noise)*finalWindScale,
					windAirresistParam,
					windVel,
					windResist);

				windVelRaw[idx] = windVel;
				windAirresistRaw[idx] = windResist;
			}

		}

	}//@namespace end of : Particle
}
#endif