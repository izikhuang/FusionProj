import os
import copy

gRawDataTypeMap =  {"AxBufferF16*":"AxFp16",\
                    "AxBufferF*":"AxFp32*",\
                    "AxBufferD*":"AxFp64*",\
                    "AxBufferInt16*":"AxInt16*",\
                    "AxBufferI*":"AxBufferHandleI32",\
                    "AxBufferInt64*":"AxInt64*",\
                    "AxBufferV3*":"AxBufferHandleV3F32",\
                    "AxScalarFieldI8*":"AxScalarFieldI8::RAWDesc",\
                    "AxScalarFieldI16*":"AxScalarFieldI16::RAWDesc",\
                    "AxScalarFieldI32*":"AxScalarFieldI32::RAWDesc",\
                    "AxScalarFieldF16*":"AxScalarFieldF16::RAWDesc",\
                    "AxScalarFieldF32*":"AxScalarFieldF32::RAWDesc",\
                    "AxScalarFieldF64*":"AxScalarFieldF64::RAWDesc",\
                    "AxFieldVector3F16*":"AxFieldVector3F16::RAWDesc",\
                    "AxFieldVector3F32*":"AxFieldVector3F32::RAWDesc",\
                    "AxFieldVector3F64*":"AxFieldVector3F64::RAWDesc",\
                    "AxVecFieldF16*":"AxVecFieldF16::RAWDesc",\
                    "AxVecFieldF32*":"AxVecFieldF32::RAWDesc",\
                    "AxVecFieldF64*":"AxVecFieldF64::RAWDesc"
            }
            

gRawMapping = {'AxBufferV3*':'AxBufferHandleV3F32',
                'AxBufferV2*':'AxVector2*',
                'AxBufferF*':'AxBufferHandleF32',
                'AxBufferI*':'AxBufferHandleI32',
                'AxVecFieldDesc':'AxVecFieldF32*',
                'AxScalarFieldDesc':'AxScalarFieldF32*'}

globalVars = {'$idx':'i'}


gGeoTypes = ['Point','Primitive','Vertex','GeoMetaData','Indices']

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

def musaReplaceFuncName(codeRet,thisKey,runtimeJson):
    codeRet = codeRet.replace(thisKey,runtimeJson['Func'])
    return codeRet

def geneFuncParamList(prmListJson,layerConstraint = -1,typeMapping=None):
    prmList = copy.copy(prmListJson)
    prmListCode = []
    for idx,prm in enumerate(prmList):
        name = prm["Name"]
        prmType = prm["Type"]
        if not typeMapping == None and prmType in typeMapping:
            prmType = typeMapping[prmType]
        cmt = prm["Comment"]
        cmtCode=""
        if len(cmt)!=0:
            cmtCode = "/*"+cmt+"*/"
        splitMark = ","
        if idx == len(prmList)-1:
            splitMark = ""
        defaultValue =""
        # print("++",name,prm["Layer"],layerConstraint)
        if layerConstraint != -1 and prm["Layer"] != layerConstraint and prm["LayerRange"] == 0:
            continue
        if "Default" in prm:
            valDefault = prm["Default"]
            if len(valDefault) !=0:
                defaultValue = " = " + prm["Default"]
        prmListCode.append(prmType + " " + name + defaultValue + splitMark + cmtCode)
    return prmListCode

def musaGeneFuncParamList(codeRet,thisKey,runtimeJson):
    prmListCode = geneFuncParamList(runtimeJson["ParamList"],0,gRawMapping)
    #print postPrmList
    prmCode = "\n".join(prmListCode)
    codeRet = codeRet.replace(thisKey,prmCode)
    return codeRet

def musaGeneFuncParamList_L1(codeRet,thisKey,runtimeJson):
    prmListCode = geneFuncParamList(runtimeJson["ParamList"],1,gRawMapping)
    #print postPrmList
    prmCode = "\n".join(prmListCode)
    codeRet = codeRet.replace(thisKey,prmCode)
    return codeRet


def toParamListMatch(prmListJson):
    prmListRet = []
    for prm in prmListJson:
        #print prm
        linkParam = prm["ExtractVar"]
        if len(linkParam) ==0:
            linkParam = prm["Name"]
        if linkParam == "blockSize":
            linkParam = "numThreads"
        if linkParam == "maxThreads":
            continue
        prmListRet.append(linkParam)
    joinStr = ","
    if len(prmListRet)>2:
        joinStr = ",\n"
    return joinStr.join(prmListRet)


#decode param 
def deCodeMethod(prm,postAdd):
    typePrm = prm["Type"]
    if typePrm in gRawMapping:
        typePrm = gRawMapping[typePrm]
    if typePrm.find("AxBuffer")!=-1 and typePrm.find('Handle')<0:
        print('typePrm:',typePrm)
        return "GetHandle"+postAdd +"()"
    return ""

def deCodeParamRawLoader(prmList,extMark=""):
    temp = "auto #VAR# = #OPT# #NAME##METHOD#;\n"
    ret=""
    for idx,prm in enumerate(prmList):
        #print(prm)
        extName = ""
        name = prm["Name"]
        methodName = deCodeMethod(prm,extMark)
        if len(methodName) !=0:
            methodName = "->"+methodName
        else:
            if name in globalVars:
                prm["ExtractVar"] = globalVars[name]
            else:
                prm["ExtractVar"] = name
            continue

        if methodName.find("Desc")!=-1:
            extName = "Desc"
        if methodName.find("DataRaw")!=-1:
            extName = "Raw"
        
        optCode = ""
        opt = prm["Optional"]
        if opt == 1:
            if prm["Type"].find("AxBuffer") != -1:
                optCode = name + "== nullptr ? nullptr : " +name+methodName
            if prm["Type"].find("Field") != -1:
                typeT = prm["Type"].replace("*",'')
                optCode = typeT+"::GetRAWDesc"+extMark+"("+prm["Name"]+")"

        baseCode = temp.replace("#VAR#",name+extName)
        ret += baseCode
        ret = ret.replace("#OPT#",optCode)

        if opt == 1:
            ret = ret.replace("#METHOD#","")
            ret = ret.replace("#NAME#","")
        else:
            ret = ret.replace("#METHOD#",methodName)
            ret = ret.replace("#NAME#",name)
            
        #print(ret)
        prm["ExtractVar"] = name + extName

    return ret

def musaGeneL2Param(codeRet,thisKey,runtimeJson):
    code = toParamListMatch(runtimeJson['ParamList'])
    codeRet = codeRet.replace(thisKey,code)
    return codeRet


def ToString(prmJson):
    realType = prmJson['Type']
    if realType in gRawDataTypeMap:
        realType = gRawDataTypeMap[realType]
    return realType +' ' +prmJson['Name']

def musaGeneShareCode(codeRet,thisKey,runtimeJson):

    #print(" mu Func Declare ")
    shareCodeTemp = runtimeJson['funcPreMACRO']+''' void #METHOD_NAME#(#PRM_LIST#)\n{
    //This block for Extend local var decleare 
    #LOCAL_VAR#\n//##IMP CODE START\n#CODE#\n//##IMP CODE END\n}\n'''
    prmCode = []
    for p in runtimeJson['ParamList']:
        prmCode.append(ToString(p))
    prmCode =  ',\n'.join(prmCode)
    shareCodeTemp = shareCodeTemp.replace('#METHOD_NAME#',runtimeJson['Func'])
    shareCodeTemp = shareCodeTemp.replace('#CODE#',runtimeJson['InlineCode'])
    shareCodeTemp = shareCodeTemp.replace('#PRM_LIST#',prmCode)
    shareCodeTemp = shareCodeTemp.replace('#LOCAL_VAR#',runtimeJson['LocalVarsCode'])

    codeRet = codeRet.replace(thisKey,shareCodeTemp)

    return codeRet


def musaDecodeCPURawParam(codeRet,thisKey,runtimeJson):
    rawParamLoadCode = deCodeParamRawLoader(runtimeJson['ParamList'])
    codeRet = codeRet.replace(thisKey,rawParamLoadCode)
    return codeRet

def musaDecodeGPURawParam(codeRet,thisKey,runtimeJson):
    rawParamLoadCode = deCodeParamRawLoader(runtimeJson['ParamList'],"Device")
    codeRet = codeRet.replace(thisKey,rawParamLoadCode)
    return codeRet

gBuffer2BaseMapping = {'AxBufferV3*':'AxVector3',
                        'AxBufferHandleI32':'AxInt32',
                        'AxBufferHandleF32':'AxFp32',
                        'AxBufferHandleV3F32':'AxVector3',
                        'AxBufferArrayHandleI32':'AxInt32',
                        'AxBufferV2*':'AxVector2',
                        'AxBufferContactHandle':'AxContact',
                        'AxBufferUCharHandle':'AxUChar',
                        'AxBufferF*':'AxFp32'}

def ExtractGeoPropMethod(parent,type):
    thisType = type
    #print('thisType:::',thisType)
    if type in gBuffer2BaseMapping:
       thisType =  gBuffer2BaseMapping[type]
    #TODO By Property
    ret = 'Get#CLASS##ARRAY_MARK#PropertyHandle<#TYPE#>'.replace("#CLASS#", parent).replace("#TYPE#",thisType)
    if 'Array' in type:
        ret = ret.replace('#ARRAY_MARK#', 'Array')
    else:
        ret = ret.replace('#ARRAY_MARK#', '')
    return ret

#SimGeo0
def evalGeoProp(geoID , name,type,parent):
    #print("PRM ++++ : ",name)
    head = 'datas->'
    methodName = 'Get'+type.replace('Ax','')+'Param'
    
    if "_" in name:
        name = name.split("_")[1]
    realNameStr = '\"'+ name.replace('$','')+'\"'
    if parent in gGeoTypes:
        head = 'datas->GetSimGeo('+str(geoID)+")->"
        methodName = ExtractGeoPropMethod(parent,type)
    if parent == 'Root':
        head = 'datas->GetSimGeo('+str(geoID)+")->"
        methodName = 'GetDesc'
        realNameStr = ''
    if parent == 'ScalarField' or parent == 'VectorField':
        head = 'datas->GetSimGeo('+str(geoID)+")->"
        methodName = 'Get'+parent+'Handle'
    method = methodName + '('+realNameStr+')'
    if parent == '':
        return name.replace('$','')
    return head + method
           
def musaCallbackParamDecode(prmListJOSN):
    layerConstraint = 0
    prmList = copy.copy(prmListJOSN)    
    prmListCode = []
    for idx,prm in enumerate(prmList):
        name = prm["Name"]
        parent = prm["ParentClass"]
        geoID = prm["GeoID"]
        #print(geoID,parent+"__"+name.replace('$','')+"__"+prm["Type"])
        splitMark = ","
        if idx == len(prmList)-1:
            splitMark = ""
        defaultValue =""
        #print("++",name,prm["Layer"],layerConstraint)
        if layerConstraint != -1 and prm["Layer"] != layerConstraint and prm["LayerRange"] == 0:
            continue
        if "Default" in prm:
            valDefault = prm["Default"]
            if len(valDefault) !=0:
                defaultValue = " = " + prm["Default"]
        prmListCode.append(evalGeoProp(geoID,name,prm["Type"],parent) + splitMark)
    return "\n".join(prmListCode)

def musaGeneL3Param(codeRet,thisKey,runtimeJson):
    callbackParamSTR = musaCallbackParamDecode(runtimeJson['ParamList'])
    codeRet = codeRet.replace(thisKey,callbackParamSTR)
    return codeRet

def musaReplacePivotFieldVar(codeRet,thisKey,runtimeJson):
    pivotName = runtimeJson['Pivot']
    codeRet = codeRet.replace(thisKey,'$'+'field_'+pivotName)
    return codeRet

def bindTasks():
    keys = {}
    keys.update({'#FUNC_NAME#':musaReplaceFuncName})        
    keys.update({'#KERNEL_PARAM_L0#':musaGeneFuncParamList})
    keys.update({'#KERNEL_PARAM_L1#':musaGeneFuncParamList_L1})        
    keys.update({'#SHARE_CODE_REPLACE#':musaGeneShareCode})        
    keys.update({'#RAW_PRM_LOAD#':musaDecodeCPURawParam})     
    keys.update({'#RAW_PRM_LOAD_DEVICE#':musaDecodeGPURawParam})     
    keys.update({'#RAW_PRM_LIST_MATCH_L2#':musaGeneL2Param})
    keys.update({'#EXTRACT_KERNEL_PARAM#':musaGeneL3Param})
    keys.update({'#PIVOT_FIELD_VAR#':musaReplacePivotFieldVar})

    return keys

def preProcess(musaKernelJson):
    prmList = musaKernelJson['ParamList']
    for prm in prmList:
        pn = prm['Name']
        pt = prm['Type']        
        if pn == "$"+"idx":
            prm["ExtractVar"]="i"
        if pt.find('AxBuffer')>=0:
            prm["ExtractVar"]=pn.replace('$','__')+'Raw'

def geneMUSAKernel(musaKernelJson,headTemp,srcTemp,cudaTemp,saveOutput):
    print("Gene Kerenel:",musaKernelJson['Func'])
    #preProcess(musaKernelJson)
    if not saveOutput.endswith("/"):
        saveOutput+="/"
    taskKeys = bindTasks()

    headCode = headTemp
    srcCode = srcTemp
    cudaCode = cudaTemp
    for key,functions in taskKeys.items():
        #print(key)
        headCode = functions(headCode,key,musaKernelJson)
        srcCode = functions(srcCode,key,musaKernelJson)
        cudaCode = functions(cudaCode,key,musaKernelJson)
    outFileName = saveOutput+musaKernelJson['Func']
    saveToFile(outFileName+".h",headCode)     #simParam
    saveToFile(outFileName+".cpp",srcCode)    #simParam
    if len(cudaCode)>0:
        saveToFile(outFileName+".cu",cudaCode)    #simParam

gIVMap = ['$Contacts',"$",'$P','PrimitiveType',"$RTriangle"]
gIVHardCodeMap = {'$RTriangle':"GetRTriangleInfo()"}
def geneIntrinsicVariableAllocationCode(jsonObject):
    print("Allocation Code")
    outCodes = []
    for prm in jsonObject['ParamList']:
        pName = prm["Name"]
        if pName not in gIVMap:
            continue

        code = ""
        dataName = pName.replace("$","")
        bufferName = dataName+"Buffer"

        if pName in gIVHardCodeMap:
            methoCodeTemp = "data.GetSimGeo(#GEO_ID#)->#METHOD#;"
            code = methoCodeTemp.replace("#METHOD#", gIVHardCodeMap[pName])
        else:
            tempCode = "auto #VAR_NAME#  = data.GetSimGeo(#GEO_ID#)->Add#GEO_NODE#Property<#RAW_DATA_TYPE#>(\"#DATA_NAME#\");"
            code = tempCode.replace("#VAR_NAME#",bufferName)
            code = code.replace("#GEO_NODE#",prm['ParentClass'])
            code = code.replace("#RAW_DATA_TYPE#",gBuffer2BaseMapping[prm['Type']])
            code = code.replace("#DATA_NAME#",dataName)
        
        
        code = code.replace("#GEO_ID#",str(prm['GeoID']))

        outCodes.append(code)
        if prm['ParentClass'] == 'Indices' and pName in jsonObject['InitStoragePool']:
                resizeCode = "    #VAR_NAME#->ResizeStorage(#NUM#);"
                resizeCode = resizeCode.replace("#NUM#", str(jsonObject['InitStoragePool'][pName]))
                resizeCode = resizeCode.replace("#VAR_NAME#", bufferName)
                resizeCode += "\n"
                outCodes.append(resizeCode)
    return "\n".join(outCodes)