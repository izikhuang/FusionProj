// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "AppleARKitFaceSupportImpl.h"
#include "AppleARKitSettings.h"
#include "AppleARKitFaceMeshConversion.h"
#include "AppleARKitConversion.h"
#include "Async/TaskGraphInterfaces.h"
#include "ARSystem.h"
#include "Misc/ConfigCacheIni.h"

DECLARE_STATS_GROUP(TEXT("AppleARKitFaceSupport"), STATGROUP_APPLEARKITFACE, STATCAT_Advanced);

#if SUPPORTS_ARKIT_1_0

static TSharedPtr<FAppleARKitAnchorData> MakeAnchorData(ARAnchor* Anchor)
{
    TSharedPtr<FAppleARKitAnchorData> NewAnchor;
    if ([Anchor isKindOfClass:[ARFaceAnchor class]])
    {
        ARFaceAnchor* FaceAnchor = (ARFaceAnchor*)Anchor;
        NewAnchor = MakeShared<FAppleARKitAnchorData>(
			FAppleARKitConversion::ToFGuid(FaceAnchor.identifier),
			FAppleARKitConversion::ToFTransform(FaceAnchor.transform),
			ToBlendShapeMap(FaceAnchor.blendShapes, FAppleARKitConversion::ToFTransform(FaceAnchor.transform)),
			ToVertexBuffer(FaceAnchor.geometry.vertices, FaceAnchor.geometry.vertexCount)
        );
        // Only convert from 16bit to 32bit once
        if (FAppleARKitAnchorData::FaceIndices.Num() == 0)
        {
            FAppleARKitAnchorData::FaceIndices = To32BitIndexBuffer(FaceAnchor.geometry.triangleIndices, FaceAnchor.geometry.triangleCount * 3);
        }
    }

    return NewAnchor;
}

#endif

FAppleARKitFaceSupport::FAppleARKitFaceSupport()
{
    // Create our LiveLink provider if the project setting is enabled
    if (GetDefault<UAppleARKitSettings>()->bEnableLiveLinkForFaceTracking)
    {
        FaceTrackingLiveLinkSubjectName = GetDefault<UAppleARKitSettings>()->DefaultFaceTrackingLiveLinkSubjectName;
#if PLATFORM_IOS
        LiveLinkSource = FAppleARKitLiveLinkSourceFactory::CreateLiveLinkSource(true);
#else
        // This should be started already, but just in case
        FAppleARKitLiveLinkSourceFactory::CreateLiveLinkRemoteListener();
#endif
    }
}

FAppleARKitFaceSupport::~FAppleARKitFaceSupport()
{
	// Should only be called durirng shutdown
	check(GIsRequestingExit);
}

#if SUPPORTS_ARKIT_1_0

ARConfiguration* FAppleARKitFaceSupport::ToARConfiguration(UARSessionConfig* SessionConfig)
{
	ARConfiguration* SessionConfiguration = nullptr;
	if (SessionConfig->GetSessionType() == EARSessionType::Face)
	{
		if (ARFaceTrackingConfiguration.isSupported == FALSE)
		{
			return nullptr;
		}
		SessionConfiguration = [ARFaceTrackingConfiguration new];
	}

	// Copy / convert properties
	SessionConfiguration.lightEstimationEnabled = SessionConfig->GetLightEstimationMode() != EARLightEstimationMode::None;
	SessionConfiguration.providesAudioData = NO;
	SessionConfiguration.worldAlignment = FAppleARKitConversion::ToARWorldAlignment(SessionConfig->GetWorldAlignment());

	return SessionConfiguration;
}

TArray<TSharedPtr<FAppleARKitAnchorData>> FAppleARKitFaceSupport::MakeAnchorData(NSArray<ARAnchor*>* Anchors, double Timestamp, uint32 FrameNumber)
{
	TArray<TSharedPtr<FAppleARKitAnchorData>> AnchorList;

	for (ARAnchor* Anchor in Anchors)
	{
		TSharedPtr<FAppleARKitAnchorData> AnchorData = ::MakeAnchorData(Anchor);
		if (AnchorData.IsValid())
		{
			AnchorList.Add(AnchorData);
		}
	}

	return AnchorList;
}

void FAppleARKitFaceSupport::PublishLiveLinkData(TSharedPtr<FAppleARKitAnchorData> Anchor, double Timestamp, uint32 FrameNumber)
{
	if (LiveLinkSource.IsValid())
	{
        LiveLinkSource->PublishBlendShapes(FaceTrackingLiveLinkSubjectName, Timestamp, FrameNumber, Anchor->BlendShapes);
	}
}

#endif
