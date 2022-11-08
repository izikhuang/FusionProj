
import hou
import json
import os
import copy

gTypeRemapping = {"Float"       : "FloatParam",\
                  "Vector2F"    : "Vector2FParam",\
                  "Ramp"        : "RampCurve32RAWData",\
                  "Vector3F"    : "Vector3FParam",\
                  "Toggle"      : "ToggleParam",\
                  "Int"         : "IntParam",\
                  "String"      : "StringParam",\
                  "IntMenuList" : "IntParam",\
                  "Vector2I"    : "Vector2IParam"}
gSrcCPPTemp = ""

def makeDefaultCode(item):
    dataType = item["DataType"]
    if dataType in gTypeRemapping == False:
        return ''
    dataType = gTypeRemapping[dataType]
    if dataType == "Vector2FParam":
        vec2Temp = "MakeVector2(#Uf,#Vf)";
        vec2Temp = vec2Temp.replace("#U",str(item["RawValue"][0]))
        vec2Temp = vec2Temp.replace("#V",str(item["RawValue"][1]))
        return vec2Temp
    if dataType == "Vector2IParam":
        vec2Temp = "MakeVector2(#U,#V)";
        vec2Temp = vec2Temp.replace("#U",str(item["RawValue"][0]))
        vec2Temp = vec2Temp.replace("#V",str(item["RawValue"][1]))
        return vec2Temp
    if dataType == "Vector3FParam":
        vec3Temp = "MakeVector3(#Uf,#Vf,#Wf)";
        vec3Temp = vec3Temp.replace("#U",str(item["RawValue"][0]))
        vec3Temp = vec3Temp.replace("#V",str(item["RawValue"][1]))
        vec3Temp = vec3Temp.replace("#W",str(item["RawValue"][2]))
        return vec3Temp
    if dataType == "StringParam":
        return "\""+str(item["RawValue"])+"\""
    if dataType == "RampCurve32RAWData":
       localBakeName = item["Name"].upper()+"_RAWDATA"
       ret =  str(item["RawValue"])
       ret = ret.replace("[","{").replace("]","}").replace(",","f,")
       ret = "AxFp32 " + localBakeName + "[32] = " + ret +";\n"
       return ret,localBakeName
    if dataType == "FloatParam":
        return str(item["RawValue"])+"f"
    return str(item["RawValue"])

def defaultParamInitAsCode(prmMap):
    retCode = ""
    prmTemp = "this->#VAR#.Set(#VAL#);\n"
    prmTempCustom = "this->#VAR# = #METHOD#(#INIT_VALS#);\n"
    #print prmMap
    for prm in prmMap:
        item = prmMap[prm]
        help = item["Help"]
        #print help.find("@NI")
        if help.find("@NI")!=-1:
            #print prm
            continue
        
        prmName = item['Name']
        prmNameUP = prmName[0].upper() + prmName[1:]
        
        if item["DataType"] == "Ramp":
            varRet = makeDefaultCode(item)
            retCode += varRet[0]
            retCode += prmTempCustom.replace("#METHOD#","MakeDefaultRampCurve32")
            retCode = retCode.replace("#VAR#",prmNameUP)
            retCode = retCode.replace("#INIT_VALS#",varRet[1])
            continue
        retCode += prmTemp.replace("#VAR#",prmNameUP)
        retCode = retCode.replace("#VAL#",makeDefaultCode(item))
        #retCode = retCode.replace("#VAL#",item['DataType'])
    return retCode

def simParamAsCode(jsonItem):
    typeName = jsonItem['DataType']
    if typeName in gTypeRemapping == False:
        return
    useTypeName = gTypeRemapping[typeName]
    retCode = ""
    prmTemp = "Ax#TYPE# #NAME#;"
    prmName = jsonItem['Name']
    prmName = prmName[0].upper() + prmName[1:]
    retCode += prmTemp.replace("#NAME#",prmName)
    retCode = retCode.replace("#TYPE#",useTypeName)
    return retCode


def varsDeserAsCode(jsonItem):
    typeName = jsonItem['DataType']
    if typeName in gTypeRemapping == False:
        return
    retCode = ""
    prmTemp = "AlphaCore::JsonHelper::Ax#TYPE#Deserilization(#VAR#, paramMap[\"#NAME#\"]);"
    prmName = jsonItem['Name']
    prmNameUP = prmName[0].upper() + prmName[1:]
    retCode += prmTemp.replace("#NAME#",prmName)
    retCode = retCode.replace("#VAR#",prmNameUP)
    retCode = retCode.replace("#TYPE#",gTypeRemapping[typeName])
    return retCode 


def make_base_dir(full_path):
    dir = os.path.dirname(full_path)
    if not os.path.exists(dir):
        os.makedirs(dir,0o777)

def saveToFile(file_name, contents):
    make_base_dir(file_name)
    fh = open(file_name, 'w')
    fh.write(contents)
    print ("Save : ",file_name)
    fh.close()

#Process #CLASS_NAME#
def msReplaceClassName(codeRet,thisKey,runtimeJson):
    oldType = runtimeJson['Type']
    typeName = oldType.replace("AxField_","AxOP_Field")

    ret = codeRet.replace(thisKey,typeName).replace('#CLASS_NAME_UPPER#',typeName.upper())

    opType=oldType.replace('_','')
    runtimeJson.update({'ClassName':typeName})
    runtimeJson.update({'OpType':opType})
    runtimeJson.update({'OpTypeName':oldType.split('_')[1]})
    runtimeJson.update({'SimParamName':opType+'SimParam'})
    
    return ret

def msReplaceOpName(codeRet,thisKey,runtimeJson):
    ret= codeRet.replace(thisKey,runtimeJson['OpType'])
    return ret.replace('#OP_TYPE_UPPER#',runtimeJson['ClassName'].upper())

#string 
#float
def msReplaceSimParamVars(codeRet,thisKey,runtimeJson):
    paramMap = runtimeJson['ParamMap']
    codes = []
    for key,value in paramMap.items():
        codes.append(simParamAsCode(value))
    varsCode = '\n'.join(codes)
    codeRet = codeRet.replace(thisKey,varsCode)
    return codeRet

def varDserCode(codeRet,thisKey,runtimeJson):
    paramMap = runtimeJson['ParamMap']
    codes = []
    for key,value in paramMap.items():
        codes.append(varsDeserAsCode(value))
    varsDeserCode = '\n'.join(codes)
    codeRet = codeRet.replace(thisKey,varsDeserCode)
    return codeRet

def varDefaultReplace(codeRet,thisKey,runtimeJson):
    paramMap = runtimeJson['ParamMap']
    codeRet = codeRet.replace(thisKey,defaultParamInitAsCode(paramMap))
    return codeRet

def bindMicroSolverFunctions():
    keys = {}
    keys.update({'#CLASS_NAME#':msReplaceClassName})        #AxOP_FieldCurlNoise
    keys.update({'#OP_TYPE#':msReplaceOpName})              #AxFieldCurlNoise
    keys.update({'#REPLACE_VARS#':msReplaceSimParamVars})   
    keys.update({'#REPLACE_VARS_DESER#':varDserCode})
    keys.update({'#PARAM_DEFAULT_REPLACE#':varDefaultReplace})
    return keys

def runMicroSolverProtocol(msDescListJson,simParamTemp,simParamCppTemp,userHeadTemp,userCppTemp,saveOutput):
    print("==================Op=====+======")
    #print codeTemplate2
    runtimeJson = {}
    if not saveOutput.endswith("/"):
        saveOutput+="/"
    taskKeys = bindMicroSolverFunctions()
    for solverDesc in msDescListJson:
        inputTemp = simParamTemp
        inputTempSrc = simParamCppTemp
        inputUserHead = userHeadTemp
        inputUserCpp = userCppTemp
        runtimeJson = solverDesc['MicroSolverDesc']
        for key,functions in taskKeys.items():
            inputTemp = functions(inputTemp,key,runtimeJson)
            inputTempSrc = functions(inputTempSrc,key,runtimeJson)
            inputUserHead = functions(inputUserHead,key,runtimeJson)
            inputUserCpp = functions(inputUserCpp,key,runtimeJson)
        outFileName = saveOutput+runtimeJson['ClassName']
        #save Code  
        saveToFile(outFileName+".ProtocolData.h",inputTemp)         #simParam
        saveToFile(outFileName+".ProtocolData.cpp",inputTempSrc)    #simParam

        srcHeadFile = outFileName+".h"
        if not os.path.isfile(srcHeadFile): 
            saveToFile(srcHeadFile,inputUserHead)
        srcCppFile = outFileName+".cpp"
        if not os.path.isfile(srcCppFile): 
            saveToFile(srcCppFile,inputUserCpp)



