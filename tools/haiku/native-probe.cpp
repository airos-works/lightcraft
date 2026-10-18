// Targeted native file-reference and panel tests; never injects keys or clicks.
#include <Application.h>
#include <Entry.h>
#include <Message.h>
#include <Messenger.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv)
{
    if (argc < 4) return 2;
    BApplication app("application/x-vnd.LightCraft-NativeProbe");
    status_t status;
    BMessenger target("application/x-vnd.jmgasper-LightCraft", atoi(argv[1]), &status);
    if (status != B_OK || !target.IsValid()) return 3;
    if (strcmp(argv[2], "refs") == 0) {
        BMessage message(B_REFS_RECEIVED);
        entry_ref ref;
        if (get_ref_for_path(argv[3], &ref) != B_OK) return 4;
        message.AddRef("refs", &ref);
        status = target.SendMessage(&message);
    } else if (strcmp(argv[2], "panel") == 0) {
        BMessage request(B_GET_PROPERTY), reply;
        request.AddSpecifier("Messenger");
        request.AddSpecifier("Window", argv[3]);
        status = target.SendMessage(&request, &reply, 1000000, 1000000);
        BMessenger panel;
        if (status != B_OK || reply.FindMessenger("result", &panel) != B_OK || !panel.IsValid()) return 5;
        printf("native panel found: %s\n", argv[3]);
        if (argc > 4 && strcmp(argv[4], "close") == 0) status = panel.SendMessage(B_QUIT_REQUESTED);
    } else return 6;
    printf("native message status: %ld\n", long(status));
    return status == B_OK ? 0 : 1;
}
