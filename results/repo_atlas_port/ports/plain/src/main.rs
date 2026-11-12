//! gitleaks: scan code, past or present, for secrets.
//! A Rust port of the Go implementation.

mod cmd;
mod config;
mod detect;
mod gojson;
mod gostr;
mod logging;
mod regexp;
mod report;
mod sources;

mod version {
    pub const DEFAULT_MSG: &str = "version is set by build process";
    pub const VERSION: &str = "version is set by build process";
}

extern "C" fn on_interrupt(_: libc::c_int) {
    let ts = chrono::Local::now().format("%-I:%M%p").to_string();
    let msg = format!("{} FTL Interrupt signal received. Exiting...\n", ts);
    unsafe {
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        libc::_exit(1);
    }
}

fn main() {
    unsafe {
        libc::signal(libc::SIGINT, on_interrupt as *const () as libc::sighandler_t);
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    cmd::execute();
}
