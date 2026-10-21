[Setup]
AppId={{919EC782-604B-4EE9-8170-81DB30AA8E3E}
AppName=BL62 Fine-tune GUI
AppVersion=1.0.3
AppPublisher=BL62
AppPublisherURL=https://github.com/Sususu-slac/BL62_Finetune_GUI
DefaultDirName={localappdata}\Programs\BL62_Finetune_GUI
DefaultGroupName=BL62 Fine-tune GUI
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=release
OutputBaseFilename=BL62_Finetune_GUI_Setup_1.0.3
SetupIconFile=assets\BL62_Finetune.ico
UninstallDisplayIcon={app}\BL62_Finetune_GUI.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: checkedonce
[Files]
Source: "dist\BL62_Finetune_GUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\BL62 Fine-tune GUI"; Filename: "{app}\BL62_Finetune_GUI.exe"
Name: "{autodesktop}\BL62 Fine-tune GUI"; Filename: "{app}\BL62_Finetune_GUI.exe"; Tasks: desktopicon; IconFilename: "{app}\_internal\assets\BL62_Finetune.ico"
[Run]
Filename: "{app}\BL62_Finetune_GUI.exe"; Description: "Launch BL62 Fine-tune GUI"; Flags: nowait postinstall skipifsilent
