/*
 * Printing through the Be print kit, bound in src/print.rs.
 *
 * Self-contained, so that the other Craft hosts can take it as it is: copy this
 * file and src/print.rs, and rename the `lc_print_` prefix in both.
 *
 * - The printers are the print server's printer folders
 *   (B_USER_PRINTERS_DIRECTORY, one folder per printer with its name, driver
 *   and transport in attributes, as the Printers preferences show them); the
 *   default printer is the print server's active printer.
 * - A printer's settings (the flattened BMessage BPrintJob takes) come from its
 *   driver: its defaults, or its page and job setup dialogs. BPrintJob's own
 *   ConfigPage and ConfigJob only reach the print server's default printer (its
 *   setup window then offers a printer menu of its own), so the driver of the
 *   printer the app's print dialog chose is asked through
 *   print_addon_server, exactly as the print server asks it.
 * - A job is a BPrintJob given those settings: each page is drawn by an
 *   off-screen view into the job (DrawView, SpoolPage) as bitmaps the caller
 *   rendered, and CommitJob hands the spool file to the print server, which
 *   moves it to the folder of the settings' `current_printer`. A small window
 *   shows the progress and can cancel the job while it is spooled.
 */

#include <Application.h>
#include <Bitmap.h>
#include <Button.h>
#include <Directory.h>
#include <Entry.h>
#include <FindDirectory.h>
#include <InterfaceDefs.h>
#include <LayoutBuilder.h>
#include <Message.h>
#include <Messenger.h>
#include <Node.h>
#include <NodeInfo.h>
#include <Path.h>
#include <PrintJob.h>
#include <Roster.h>
#include <Screen.h>
#include <StatusBar.h>
#include <String.h>
#include <View.h>
#include <Window.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <string>
#include <vector>


// The print server's protocol (headers/private/print/pr_server.h).
static const char* kPrintServerSignature = "application/x-vnd.Be-PSRV";
static const uint32 kGetActivePrinter = 'pgcp';
static const uint32 kMakePrinterActiveQuietly = 'pmaq';
static const char* kPrinterFileType = "application/x-vnd.Be.printer";

// print_addon_server's protocol (headers/private/print/
// PrintAddOnServerProtocol.h): it loads a printer driver add-on and calls one
// of its hooks for the printer folder it is given.
static const char* kAddOnServerSignature
	= "application/x-vnd.haiku-print-addon-server";
static const uint32 kAddOnConfigPage = 'PScp';
static const uint32 kAddOnConfigJob = 'PScj';
static const uint32 kAddOnDefaultSettings = 'PSds';

// The driver a stored settings message belongs to: another driver's settings
// (the printer was set up again) are not used.
static const char* kDriverField = "craft:driver";


// #pragma mark - helpers


static BString
read_attr(BNode& node, const char* name)
{
	BString value;
	if (node.ReadAttrString(name, &value) != B_OK)
		value = "";
	return value;
}


static BMessenger
print_server()
{
	BMessenger messenger(kPrintServerSignature);
	if (!messenger.IsValid()) {
		// The print server starts on demand, as BPrintJob would need it.
		team_id team = -1;
		if (be_roster->Launch(kPrintServerSignature, (BMessage*)NULL, &team)
				== B_OK || team >= 0)
			messenger = BMessenger(kPrintServerSignature, team);
	}
	return messenger;
}


static BString
active_printer()
{
	BString name;
	BMessenger server = print_server();
	if (!server.IsValid())
		return name;
	BMessage request(kGetActivePrinter);
	BMessage reply;
	if (server.SendMessage(&request, &reply, 2000000, 5000000) == B_OK)
		reply.FindString("printer_name", &name);
	return name;
}


static bool
printer_folder(const char* printer, BPath& folder, BString& driver)
{
	if (printer == NULL || printer[0] == '\0' || strchr(printer, '/') != NULL)
		return false;
	if (find_directory(B_USER_PRINTERS_DIRECTORY, &folder) != B_OK
		|| folder.Append(printer) != B_OK)
		return false;
	BNode node(folder.Path());
	if (node.InitCheck() != B_OK)
		return false;
	driver = read_attr(node, "Driver Name");
	return driver.Length() > 0;
}


static void
set_int32(BMessage& message, const char* name, int32 value)
{
	if (message.ReplaceInt32(name, value) != B_OK) {
		message.RemoveName(name);
		message.AddInt32(name, value);
	}
}


static void
set_bool(BMessage& message, const char* name, bool value)
{
	if (message.ReplaceBool(name, value) != B_OK) {
		message.RemoveName(name);
		message.AddBool(name, value);
	}
}


static void
set_string(BMessage& message, const char* name, const char* value)
{
	message.RemoveName(name);
	message.AddString(name, value);
}


// A setting the drivers store as int64 (libprint) or int32 (Preview).
static int32
find_number(const BMessage& message, const char* name, int32 fallback)
{
	int64 v64;
	int32 v32;
	float vf;
	if (message.FindInt64(name, &v64) == B_OK)
		return (int32)v64;
	if (message.FindInt32(name, &v32) == B_OK)
		return v32;
	if (message.FindFloat(name, &vf) == B_OK)
		return (int32)vf;
	return fallback;
}


static void*
flatten(const BMessage& message, size_t* length)
{
	*length = 0;
	ssize_t size = message.FlattenedSize();
	if (size <= 0)
		return NULL;
	void* data = malloc(size);
	if (data == NULL)
		return NULL;
	if (message.Flatten((char*)data, size) != B_OK) {
		free(data);
		return NULL;
	}
	*length = size;
	return data;
}


static std::string
clean(const BString& text)
{
	std::string out(text.String());
	for (char& c : out) {
		if (c == '\t' || c == '\n' || c == '\r')
			c = ' ';
	}
	return out;
}


// #pragma mark - the job's windows


// Draws the page's bitmaps where the caller placed them, for DrawView.
class PrintPageView : public BView {
public:
	struct Tile {
		BBitmap* bitmap;
		BRect destination;
	};

	PrintPageView()
		:
		BView(BRect(0, 0, 1, 1), "page", B_FOLLOW_NONE, B_WILL_DRAW),
		fOptions(0)
	{
		// Nothing but the bitmaps goes into the job: the paper is white.
		SetViewColor(B_TRANSPARENT_COLOR);
	}

	~PrintPageView()
	{
		Clear();
	}

	void Clear()
	{
		for (Tile& tile : fTiles)
			delete tile.bitmap;
		fTiles.clear();
	}

	void Add(BBitmap* bitmap, BRect destination)
	{
		fTiles.push_back({bitmap, destination});
	}

	void SetOptions(uint32 options)
	{
		fOptions = options;
	}

	virtual void Draw(BRect update)
	{
		for (const Tile& tile : fTiles) {
			if (tile.destination.Intersects(update))
				DrawBitmap(tile.bitmap, tile.bitmap->Bounds(), tile.destination,
					fOptions);
		}
	}

private:
	std::vector<Tile> fTiles;
	uint32 fOptions;
};


static const uint32 kProgressCancel = 'Pcnc';
static const uint32 kProgressUpdate = 'Pupd';


// "Printing “title”", the pages spooled so far and a Cancel button.
class PrintProgressWindow : public BWindow {
public:
	PrintProgressWindow(const char* title, std::atomic<bool>* cancelled)
		:
		BWindow(BRect(0, 0, 360, 80), "Printing", B_TITLED_WINDOW_LOOK,
			B_NORMAL_WINDOW_FEEL,
			B_NOT_RESIZABLE | B_NOT_ZOOMABLE | B_NOT_MINIMIZABLE
				| B_AUTO_UPDATE_SIZE_LIMITS | B_ASYNCHRONOUS_CONTROLS),
		fCancelled(cancelled)
	{
		BString label("Printing “");
		label << title << "”";
		fStatus = new BStatusBar("status", label.String());
		fStatus->SetExplicitMinSize(BSize(320, B_SIZE_UNSET));
		fCancel = new BButton("cancel", "Cancel", new BMessage(kProgressCancel));
		BLayoutBuilder::Group<>(this, B_VERTICAL, B_USE_DEFAULT_SPACING)
			.SetInsets(B_USE_WINDOW_SPACING)
			.Add(fStatus)
			.AddGroup(B_HORIZONTAL)
				.AddGlue()
				.Add(fCancel)
			.End();
		CenterOnScreen();
	}

	virtual void MessageReceived(BMessage* message)
	{
		switch (message->what) {
			case kProgressCancel:
				fCancelled->store(true);
				fCancel->SetEnabled(false);
				fStatus->SetTrailingText("Cancelling…");
				break;
			case kProgressUpdate:
			{
				int32 done = 0;
				int32 total = 1;
				message->FindInt32("done", &done);
				message->FindInt32("total", &total);
				fStatus->SetMaxValue(total > 0 ? total : 1);
				fStatus->SetTo(done);
				if (!fCancelled->load()) {
					BString text;
					text << "Page " << (done < total ? done + 1 : total)
						<< " of " << total;
					fStatus->SetTrailingText(text.String());
				}
				break;
			}
			default:
				BWindow::MessageReceived(message);
		}
	}

	virtual bool QuitRequested()
	{
		// The close box cancels; the job closes the window itself.
		fCancelled->store(true);
		return false;
	}

private:
	BStatusBar* fStatus;
	BButton* fCancel;
	std::atomic<bool>* fCancelled;
};


struct LcPrintJob {
	BPrintJob* job;
	BWindow* window;
	PrintPageView* view;
	PrintProgressWindow* progress;
	std::atomic<bool> cancelled;
};


static void
close_job_windows(LcPrintJob* job)
{
	if (job->progress != NULL && job->progress->Lock())
		job->progress->Quit();
	job->progress = NULL;
	if (job->window != NULL && job->window->Lock())
		job->window->Quit();
	job->window = NULL;
	job->view = NULL;
}


extern "C" {


// Keep in step with `RawOptions` in src/print.rs.
struct LcPrintOptions {
	int32_t copies;
	int32_t collate;
	// -1 the printer's default, 0 one-sided, 1 long edge, 2 short edge.
	int32_t duplex;
	int32_t grayscale;
	// The pages the job will have: its page range is all of them.
	int32_t pages;
};

// Keep in step with `RawJobInfo` in src/print.rs. Rectangles are left, top,
// right, bottom in points (1/72 inch).
struct LcPrintJobInfo {
	float paper[4];
	float printable[4];
	int32_t xres;
	int32_t yres;
	int32_t first_page;
	int32_t last_page;
	int32_t copies;
};

// Keep in step with `RawTile` in src/print.rs: a bitmap of `width` x `height`
// pixels, B_RGB32 (B, G, R, A bytes) or with `gray` B_GRAY8, drawn into
// `destination` (left, top, right, bottom in the page's points).
struct LcPrintTile {
	const uint8_t* pixels;
	int32_t width;
	int32_t height;
	int32_t stride;
	int32_t gray;
	float destination[4];
};


void
lc_print_free(void* data)
{
	free(data);
}


// The printers, one per line: name, driver, transport, 1 for the default
// printer (else 0) and comments, separated by tabs. malloc'd.
char*
lc_print_printers()
{
	std::string out;
	BPath path;
	if (find_directory(B_USER_PRINTERS_DIRECTORY, &path) == B_OK) {
		BString active = active_printer();
		BDirectory dir(path.Path());
		BEntry entry;
		while (dir.GetNextEntry(&entry) == B_OK) {
			if (!entry.IsDirectory())
				continue;
			BNode node(&entry);
			char type[B_MIME_TYPE_LENGTH];
			BNodeInfo info(&node);
			if (info.GetType(type) != B_OK || strcmp(type, kPrinterFileType) != 0)
				continue;
			// The folder's name is the printer's name, which jobs name.
			char name[B_FILE_NAME_LENGTH];
			if (entry.GetName(name) != B_OK)
				continue;
			BString driver = read_attr(node, "Driver Name");
			if (driver.Length() == 0)
				continue;
			out += clean(name);
			out += '\t';
			out += clean(driver);
			out += '\t';
			out += clean(read_attr(node, "transport"));
			out += '\t';
			out += active == name ? "1" : "0";
			out += '\t';
			out += clean(read_attr(node, "Comments"));
			out += '\n';
		}
	}
	return strdup(out.c_str());
}


// The settings of `printer` (flattened, malloc'd; NULL with `*status` on
// failure, B_CANCELED when a dialog was cancelled). `mode` 0: the driver's
// defaults (B_UNSUPPORTED when it has none); 1: its page setup dialog and 2: its job setup dialog, both starting
// from `settings`; 3: `settings` as they are. `options`, when given, are
// applied first.
void*
lc_print_settings(const char* printer, int32_t mode, const void* settings,
	size_t settingsLength, const LcPrintOptions* options, size_t* length,
	int32_t* status)
{
	*length = 0;
	*status = B_OK;
	BPath folder;
	BString driver;
	if (!printer_folder(printer, folder, driver)) {
		*status = B_ENTRY_NOT_FOUND;
		return NULL;
	}

	BMessage message;
	if (settings != NULL && settingsLength > 0 && mode != 0) {
		status_t result = message.Unflatten((const char*)settings);
		if (result != B_OK) {
			*status = result;
			return NULL;
		}
		BString stored;
		if (message.FindString(kDriverField, &stored) == B_OK
			&& stored != driver) {
			// Settings of the printer's former driver: start again.
			message.MakeEmpty();
			mode = mode == 3 ? 0 : mode;
		}
	}

	if (mode == 0 || (mode != 3 && message.IsEmpty())) {
		BMessage request(kAddOnDefaultSettings);
		request.AddString("driver", driver.String());
		request.AddString("folder", folder.Path());
		team_id team = -1;
		status_t result = be_roster->Launch(kAddOnServerSignature,
			(BMessage*)NULL, &team);
		if (result != B_OK) {
			*status = result;
			return NULL;
		}
		BMessenger server(kAddOnServerSignature, team);
		BMessage reply;
		result = server.SendMessage(&request, &reply, 30000000, 60000000);
		server.SendMessage(B_QUIT_REQUESTED);
		int32 code = B_ERROR;
		if (result == B_OK && reply.FindInt32("status", &code) == B_OK) {
			// Only some drivers (Preview) have defaults: libprint's (PostScript,
			// PCL, Gutenprint) are set up through their page setup dialog.
			result = code == B_OK ? B_OK : B_UNSUPPORTED;
		}
		if (result == B_OK && reply.FindMessage("settings", &message) != B_OK)
			result = B_UNSUPPORTED;
		// A setup dialog starts from nothing as well.
		if (result != B_OK && (mode == 0 || result != B_UNSUPPORTED)) {
			*status = result;
			return NULL;
		}
	}

	set_string(message, "current_printer", printer);
	set_string(message, kDriverField, driver.String());
	if (options != NULL) {
		set_int32(message, "copies", options->copies > 0 ? options->copies : 1);
		// libprint's drivers (PostScript, PCL, Gutenprint, LIPS): collation,
		// every page (the app has picked them), two-sided printing (which
		// they use only where the printer has it) and monochrome; others
		// ignore these fields.
		set_bool(message, "JJJJ_collate", options->collate != 0);
		set_int32(message, "first_page", 1);
		set_int32(message, "last_page", options->pages > 0 ? options->pages : 1);
		set_int32(message, "JJJJ_page_selection", 0);	// JobData::kAllPages
		if (options->duplex >= 0) {
			// JobData::kSimplex, kDuplex; kLongEdgeLeft, kShortEdgeTop.
			set_int32(message, "JJJJ_print_style", options->duplex == 0 ? 0 : 1);
			if (options->duplex > 0)
				set_int32(message, "JJJJ_binding_location",
					options->duplex == 2 ? 2 : 0);
		}
		if (options->grayscale)
			set_int32(message, "JJJJ_color", 1);	// JobData::kMonochrome
	}

	if (mode == 1 || mode == 2) {
		BMessage request(mode == 1 ? kAddOnConfigPage : kAddOnConfigJob);
		request.AddString("driver", driver.String());
		request.AddString("folder", folder.Path());
		request.AddMessage("settings", &message);
		team_id team = -1;
		status_t result = be_roster->Launch(kAddOnServerSignature,
			(BMessage*)NULL, &team);
		if (result != B_OK) {
			*status = result;
			return NULL;
		}
		BMessenger server(kAddOnServerSignature, team);
		BMessage reply;
		// The user takes as long as they need.
		result = server.SendMessage(&request, &reply, 30000000,
			B_INFINITE_TIMEOUT);
		server.SendMessage(B_QUIT_REQUESTED);
		int32 code = B_ERROR;
		if (result == B_OK && reply.FindInt32("status", &code) == B_OK)
			result = code;
		BMessage configured;
		if (result == B_OK
			&& reply.FindMessage("settings", &configured) != B_OK)
			result = B_BAD_DATA;
		if (result != B_OK) {
			// The drivers return an error when their dialog is cancelled.
			*status = B_CANCELED;
			return NULL;
		}
		message = configured;
		set_string(message, "current_printer", printer);
		set_string(message, kDriverField, driver.String());
	}

	void* out = flatten(message, length);
	if (out == NULL)
		*status = B_NO_MEMORY;
	return out;
}


// The system's print setup, for apps without a printer list of their own:
// BPrintJob's ConfigPage and ConfigJob, which the print server shows with its
// printer menu, then the chosen printer's page and job setup. Starts from
// `settings` when given. Flattened and malloc'd; NULL with `*status`
// (B_CANCELED when cancelled).
void*
lc_print_system_setup(const void* settings, size_t settingsLength,
	size_t* length, int32_t* status)
{
	*length = 0;
	*status = B_OK;
	BPrintJob job("setup");
	if (settings != NULL && settingsLength > 0) {
		BMessage* message = new BMessage;
		if (message->Unflatten((const char*)settings) == B_OK)
			job.SetSettings(message);
		else
			delete message;
	}
	if (job.ConfigPage() != B_OK || job.ConfigJob() != B_OK) {
		*status = B_CANCELED;
		return NULL;
	}
	BMessage* chosen = job.Settings();
	if (chosen == NULL) {
		*status = B_ERROR;
		return NULL;
	}
	BString printer;
	BPath folder;
	BString driver;
	if (chosen->FindString("current_printer", &printer) == B_OK
		&& printer_folder(printer.String(), folder, driver))
		set_string(*chosen, kDriverField, driver.String());
	void* out = flatten(*chosen, length);
	delete chosen;
	if (out == NULL)
		*status = B_NO_MEMORY;
	return out;
}


// Starts a job of `settings` (from lc_print_settings) and fills `info` with its
// paper. NULL with `*status` on failure.
LcPrintJob*
lc_print_begin(const void* settings, size_t settingsLength, const char* title,
	LcPrintJobInfo* info, int32_t* status)
{
	*status = B_OK;
	BMessage* message = new BMessage;
	if (settings == NULL || message->Unflatten((const char*)settings) != B_OK) {
		delete message;
		*status = B_BAD_DATA;
		return NULL;
	}
	BString printer;
	message->FindString("current_printer", &printer);
	// BPrintJob writes the spool file into the default printer's folder before
	// the print server moves it to `current_printer`'s: without a default
	// printer it has nowhere to go. Where none is set (printers were added but
	// none was made the default), the one printed to becomes the default.
	if (active_printer().Length() == 0 && printer.Length() > 0) {
		BMessage select(kMakePrinterActiveQuietly);
		select.AddString("printer", printer.String());
		BMessage reply;
		print_server().SendMessage(&select, &reply, 2000000, 5000000);
	}
	if (active_printer().Length() == 0) {
		delete message;
		*status = B_NO_INIT;
		return NULL;
	}

	BRect paper(0, 0, 611, 791);
	BRect printable;
	message->FindRect("paper_rect", &paper);
	if (message->FindRect("printable_rect", &printable) != B_OK)
		printable = paper;
	info->paper[0] = paper.left;
	info->paper[1] = paper.top;
	info->paper[2] = paper.right;
	info->paper[3] = paper.bottom;
	info->printable[0] = printable.left;
	info->printable[1] = printable.top;
	info->printable[2] = printable.right;
	info->printable[3] = printable.bottom;
	info->xres = find_number(*message, "xres", 300);
	info->yres = find_number(*message, "yres", info->xres);
	info->first_page = find_number(*message, "first_page", 1);
	info->last_page = find_number(*message, "last_page", -1);
	info->copies = find_number(*message, "copies", 1);

	LcPrintJob* job = new LcPrintJob;
	job->cancelled.store(false);
	job->job = new BPrintJob(title != NULL && title[0] ? title : "Document");
	job->job->SetSettings(message);
	job->job->BeginJob();
	if (!job->job->CanContinue()) {
		delete job->job;
		delete job;
		*status = B_ERROR;
		return NULL;
	}

	// The page is drawn by a view of a window that is never shown.
	job->window = new BWindow(BRect(0, 0, 10, 10), "print",
		B_NO_BORDER_WINDOW_LOOK, B_NORMAL_WINDOW_FEEL,
		B_AVOID_FOCUS | B_NOT_MOVABLE | B_NOT_RESIZABLE);
	job->view = new PrintPageView;
	job->window->AddChild(job->view);
	job->window->Hide();
	job->window->Show();

	job->progress = new PrintProgressWindow(
		title != NULL && title[0] ? title : "Document", &job->cancelled);
	job->progress->Show();
	return job;
}


// The progress window's count: `done` pages of `total` spooled.
void
lc_print_progress(LcPrintJob* job, int32_t done, int32_t total)
{
	if (job == NULL || job->progress == NULL)
		return;
	BMessage update(kProgressUpdate);
	update.AddInt32("done", done);
	update.AddInt32("total", total);
	BMessenger(job->progress).SendMessage(&update);
}


// Whether the progress window's Cancel was pressed.
int32_t
lc_print_cancelled(LcPrintJob* job)
{
	return job != NULL && job->cancelled.load() ? 1 : 0;
}


// Adds a page of `width` x `height` points made of `count` tiles, placed at
// `x`, `y` points from the printable rectangle's top left; `filter` smooths
// bitmaps the printer draws larger than they are. B_OK, or an error when the
// job can't continue.
int32_t
lc_print_page(LcPrintJob* job, const LcPrintTile* tiles, int32_t count,
	float width, float height, float x, float y, int32_t filter)
{
	if (job == NULL || job->window == NULL)
		return B_BAD_VALUE;
	if (!job->window->Lock())
		return B_ERROR;
	job->view->Clear();
	job->view->ResizeTo(width, height);
	job->view->SetOptions(filter ? B_FILTER_BITMAP_BILINEAR : 0);
	status_t result = B_OK;
	for (int32 i = 0; i < count && result == B_OK; i++) {
		const LcPrintTile& tile = tiles[i];
		if (tile.width <= 0 || tile.height <= 0)
			continue;
		BBitmap* bitmap = new BBitmap(BRect(0, 0, tile.width - 1,
			tile.height - 1), tile.gray ? B_GRAY8 : B_RGB32);
		if (bitmap->InitCheck() != B_OK) {
			delete bitmap;
			result = B_NO_MEMORY;
			break;
		}
		int32 rowBytes = tile.width * (tile.gray ? 1 : 4);
		uint8* bits = (uint8*)bitmap->Bits();
		for (int32 row = 0; row < tile.height; row++) {
			memcpy(bits + row * bitmap->BytesPerRow(),
				tile.pixels + (size_t)row * tile.stride, rowBytes);
		}
		job->view->Add(bitmap, BRect(tile.destination[0], tile.destination[1],
			tile.destination[2], tile.destination[3]));
	}
	job->window->Unlock();
	if (result != B_OK)
		return result;

	job->job->DrawView(job->view, BRect(0, 0, width, height), BPoint(x, y));
	job->job->SpoolPage();

	// DrawView has recorded the page (its picture is complete once it
	// returns), so the bitmaps can go.
	if (job->window->Lock()) {
		job->view->Clear();
		job->window->Unlock();
	}
	return job->job->CanContinue() ? B_OK : B_ERROR;
}


// Hands the job to the print server and frees it.
int32_t
lc_print_commit(LcPrintJob* job)
{
	if (job == NULL)
		return B_BAD_VALUE;
	close_job_windows(job);
	job->job->CommitJob();
	delete job->job;
	delete job;
	return B_OK;
}


// Drops the job (its spool file is removed) and frees it.
void
lc_print_cancel(LcPrintJob* job)
{
	if (job == NULL)
		return;
	close_job_windows(job);
	job->job->CancelJob();
	delete job->job;
	delete job;
}


// Whether Shift is held: the app's print button then opens the driver's setup
// dialogs first.
int32_t
lc_print_shift_down()
{
	return (modifiers() & B_SHIFT_KEY) != 0 ? 1 : 0;
}


}
