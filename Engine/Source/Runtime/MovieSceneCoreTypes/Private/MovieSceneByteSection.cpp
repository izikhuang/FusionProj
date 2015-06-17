// Copyright 1998-2015 Epic Games, Inc. All Rights Reserved.

#include "MovieSceneCoreTypesPCH.h"
#include "MovieSceneByteSection.h"


UMovieSceneByteSection::UMovieSceneByteSection( const FObjectInitializer& ObjectInitializer )
	: Super( ObjectInitializer )
{
}

uint8 UMovieSceneByteSection::Eval( float Position ) const
{
	return !!ByteCurve.Evaluate(Position);
}

void UMovieSceneByteSection::MoveSection( float DeltaPosition, TSet<FKeyHandle>& KeyHandles )
{
	Super::MoveSection( DeltaPosition, KeyHandles );

	ByteCurve.ShiftCurve(DeltaPosition, KeyHandles);
}

void UMovieSceneByteSection::DilateSection( float DilationFactor, float Origin, TSet<FKeyHandle>& KeyHandles )
{
	Super::DilateSection(DilationFactor, Origin, KeyHandles);
	
	ByteCurve.ScaleCurve(Origin, DilationFactor, KeyHandles);
}

void UMovieSceneByteSection::GetKeyHandles(TSet<FKeyHandle>& KeyHandles) const
{
	for (auto It(ByteCurve.GetKeyHandleIterator()); It; ++It)
	{
		float Time = ByteCurve.GetKeyTime(It.Key());
		if (IsTimeWithinSection(Time))
			KeyHandles.Add(It.Key());
	}
}

void UMovieSceneByteSection::AddKey( float Time, uint8 Value )
{
	Modify();
	ByteCurve.UpdateOrAddKey(Time, Value ? 1 : 0);
}

bool UMovieSceneByteSection::NewKeyIsNewData(float Time, uint8 Value) const
{
	return ByteCurve.GetNumKeys() == 0 || Eval(Time) != Value;
}
