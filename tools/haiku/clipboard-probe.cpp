// Native clipboard interoperability fixture. Never prints clipboard contents.
#include <Application.h>
#include <Bitmap.h>
#include <Clipboard.h>
#include <File.h>
#include <Message.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    BApplication app("application/x-vnd.LightCraft-ClipboardTest");
    if (!be_clipboard->Lock()) return 3;
    BMessage* clip = be_clipboard->Data();
    status_t status = B_BAD_VALUE;
    if (argc == 3 && strcmp(argv[1], "backup") == 0) {
        BFile file(argv[2], B_WRITE_ONLY | B_CREATE_FILE | B_FAIL_IF_EXISTS);
        status = file.InitCheck();
        if (status == B_OK) status = file.SetPermissions(0600);
        if (status == B_OK) status = clip->Flatten(&file);
    } else if (argc == 3 && strcmp(argv[1], "restore") == 0) {
        BFile file(argv[2], B_READ_ONLY);
        BMessage backup;
        status = backup.Unflatten(&file);
        if (status == B_OK) {
            be_clipboard->Clear();
            *clip = backup;
            status = be_clipboard->Commit();
        }
        if (status == B_OK) unlink(argv[2]);
    } else if (argc == 3 && strcmp(argv[1], "text") == 0) {
        be_clipboard->Clear();
        status = clip->AddData("text/plain", B_MIME_TYPE, argv[2], strlen(argv[2]));
        if (status == B_OK) status = be_clipboard->Commit();
    } else if (argc == 3 && strcmp(argv[1], "check-text") == 0) {
        const void* bytes = NULL;
        ssize_t size = 0;
        status = clip->FindData("text/plain", B_MIME_TYPE, &bytes, &size);
        if (status == B_OK && (size != (ssize_t)strlen(argv[2]) || memcmp(bytes, argv[2], size) != 0)) status = B_ERROR;
        printf("native clipboard text matches: %s\n", status == B_OK ? "true" : "false");
    } else if (strcmp(argv[1], "native") == 0) {
        BBitmap bitmap(BRect(0, 0, 63, 31), B_RGBA32);
        status = bitmap.InitCheck();
        if (status == B_OK) {
            for (int y = 0; y < 32; ++y) {
                uint8* row = (uint8*)bitmap.Bits() + y * bitmap.BytesPerRow();
                for (int x = 0; x < 64; ++x) {
                    row[4*x] = 180; row[4*x+1] = 50 + y*4;
                    row[4*x+2] = 30 + x*3; row[4*x+3] = 255;
                }
            }
            BMessage archive;
            status = bitmap.Archive(&archive);
            if (status == B_OK) {
                be_clipboard->Clear();
                status = clip->AddMessage("image/bitmap", &archive);
                if (status == B_OK) status = be_clipboard->Commit();
            }
        }
    } else if (strcmp(argv[1], "check") == 0) {
        const void* bytes = NULL;
        ssize_t size = 0;
        BMessage archive;
        BRect bounds;
        bool png = clip->FindData("image/png", B_MIME_TYPE, &bytes, &size) == B_OK;
        bool native = clip->FindMessage("image/bitmap", &archive) == B_OK
            && archive.FindRect("_frame", &bounds) == B_OK;
        printf("{\"png\":%s,\"nativeBitmap\":%s,\"width\":%.0f,\"height\":%.0f}\n",
            png ? "true" : "false", native ? "true" : "false",
            native ? bounds.Width()+1 : 0, native ? bounds.Height()+1 : 0);
        status = png && native ? B_OK : B_ERROR;
    }
    be_clipboard->Unlock();
    return status == B_OK ? 0 : 1;
}
