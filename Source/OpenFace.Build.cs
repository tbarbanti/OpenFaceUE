// Some copyright should be here...

using UnrealBuildTool;
using System.IO;

public class OpenFace : ModuleRules
{
    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../ThirdParty/")); }
    }

	public OpenCV(TargetInfo Target)
	{
        // Startard Module Dependencies
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "RHI", "RenderCore" });
		PrivateDependencyModuleNames.AddRange(new string[] { "CoreUObject", "Engine", "Slate", "SlateCore" });

        // Start OpenCV linking here!
        bool isLibrarySupported = false;

        bEnableUndefinedIdentifierWarnings = false;

        // Create  Path 
        string OpenCVPath = Path.Combine(ThirdPartyPath, "OpenCV");
        string OpenBLASPath = Path.Combine(ThirdPartyPath, "OpenBLAS");
        string boostPath = Path.Combine(ThirdPartyPath, "boost");
        string dlibPath = Path.Combine(ThirdPartyPath, "dlib");
        string tbbPath = Path.Combine(ThirdPartyPath, "tbb");
        string FaceAnalyserPath = Path.Combine(ThirdPartyPath, "FaceAnalyser");
        string GazeAnalyserPath = Path.Combine(ThirdPartyPath, "GazeAnalyser");
        string LandmarkDetectorPath = Path.Combine(ThirdPartyPath, "LandmarkDetector");

        // Get Library Path 
        string LibPath = "";
        bool isdebug = Target.Configuration == UnrealTargetConfiguration.Debug && BuildConfiguration.bDebugBuildsActuallyUseDebugCRT;
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            LibPath = Path.Combine(OpenCVPath, "Libraries", "Win64");
            isLibrarySupported = true;
        }
        else if (Target.Platform == UnrealTargetPlatform.Win32)
        {
            // TODO: add OpenCV binaries for Win32
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            // TODO: add OpenCV binaries for Mac
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            // TODO: add OpenCV binaries for Linux
        }
        else
        {
            string Err = string.Format("{0} dedicated server is made to depend on {1}. We want to avoid this, please correct module dependencies.", Target.Platform.ToString(), this.ToString()); System.Console.WriteLine(Err);
        }

        if (isLibrarySupported)
        {
            //Add Include path 
            PublicIncludePaths.AddRange(new string[] { Path.Combine(OpenCVPath, "Includes") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(boostPath, "boost") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(dlibPath, "include") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(OpenBLASPath, "include") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(tbbPath, "include") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(FaceAnalyserPath, "include") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(GazeAnalyserPath, "include") });
            PublicIncludePaths.AddRange(new string[] { Path.Combine(LandmarkDetectorPath, "include") });

            // Add Library Path 
            PublicLibraryPaths.Add(LibPath);

            //Add Static Libraries
            PublicAdditionalLibraries.Add("opencv_world310.lib");
            PublicAdditionalLibraries.Add("dlib.lib");
            PublicAdditionalLibraries.Add("FaceAnalyser.lib");
            PublicAdditionalLibraries.Add("GazeAnalyser.lib");
            PublicAdditionalLibraries.Add("LandmarkDetector.lib");
            PublicAdditionalLibraries.Add("libboost_filesystem-vc140-mt-1_60.lib");
            PublicAdditionalLibraries.Add("libboost_filesystem-vc140-mt-gd-1_60.lib");
            PublicAdditionalLibraries.Add("libboost_system-vc140-mt-1_60.lib");
            PublicAdditionalLibraries.Add("libboost_system-vc140-mt-gd-1_60.lib");
            PublicAdditionalLibraries.Add("tbb.lib");
            PublicAdditionalLibraries.Add("tbbmalloc.lib");
            PublicAdditionalLibraries.Add("tbbmalloc_proxy.lib");
            PublicAdditionalLibraries.Add("tbbmalloc_s.lib");
            PublicAdditionalLibraries.Add("tbbproxy.lib");
            PublicAdditionalLibraries.Add("tbb_debug.lib");
            PublicAdditionalLibraries.Add("tbb_preview.lib");
            PublicAdditionalLibraries.Add("libopenblas.dll.a");

            //Add Dynamic Libraries
            PublicDelayLoadDLLs.Add("opencv_world310.dll");
            PublicDelayLoadDLLs.Add("opencv_ffmpeg310_64.dll");
            PublicDelayLoadDLLs.Add("libgcc_s_seh-1.dll");
            PublicDelayLoadDLLs.Add("libgfortran-3.dll");
            PublicDelayLoadDLLs.Add("libopenblas.dll");
            PublicDelayLoadDLLs.Add("libquadmath-0.dll");
            PublicDelayLoadDLLs.Add("tbb.dll");
            PublicDelayLoadDLLs.Add("tbbmalloc.dll");
            PublicDelayLoadDLLs.Add("tbbmalloc_proxy.dll");
            PublicDelayLoadDLLs.Add("tbb_debug.dll");
            PublicDelayLoadDLLs.Add("tbb_preview.dll");

        }

        Definitions.Add(string.Format("WITH_OPENCV_BINDING={0}", isLibrarySupported ? 1 : 0));
        Definitions.Add("_WINDOWS");
        Definitions.Add("WIN64=1");
        Definitions.Add("_WIN64=1");
        Definitions.Add("TBB_ARCH_PLATFORM=intel64/vc14");
        Definitions.Add("TBB_TARGET_ARCH=intel64");
        Definitions.Add("DLIB_NO_GUI_SUPPORT");
        Definitions.Add("DLIB_HAVE_SSE2");
        Definitions.Add("DLIB_HAVE_SSE3");
        Definitions.Add("DLIB_HAVE_SSE41");
        Definitions.Add("DLIB_NO_GUI_SUPPORT");
        
    }
}
