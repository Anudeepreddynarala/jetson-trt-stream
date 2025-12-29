use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use sysinfo::System;

use crate::protocol::{Command, CommandResponse};

pub struct AppState {
    pub start_time: Instant,
    pub packets_sent: Arc<RwLock<u64>>,
    pub system: Arc<RwLock<System>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            packets_sent: Arc::new(RwLock::new(0)),
            system: Arc::new(RwLock::new(System::new_all())),
        }
    }
}

pub struct TcpCommandServer {
    listener: TcpListener,
    state: Arc<AppState>,
}

impl TcpCommandServer {
    pub async fn new(bind_addr: &str, state: Arc<AppState>) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr).await?;
        log::info!("TCP command server listening on {}", bind_addr);

        Ok(Self { listener, state })
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            let (socket, addr) = self.listener.accept().await?;
            log::info!("TCP client connected: {}", addr);

            let state = self.state.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_client(socket, state).await {
                    log::error!("Client handler error: {}", e);
                }
            });
        }
    }
}

async fn handle_client(socket: TcpStream, state: Arc<AppState>) -> Result<()> {
    let (reader, mut writer) = socket.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;

        if n == 0 {
            log::info!("TCP client disconnected");
            break;
        }

        // Parse command
        let cmd: Command = match serde_json::from_str(&line) {
            Ok(c) => c,
            Err(e) => {
                let err_resp = CommandResponse::error(format!("Invalid JSON: {}", e));
                let response = serde_json::to_string(&err_resp)? + "\n";
                writer.write_all(response.as_bytes()).await?;
                continue;
            }
        };

        // Handle command
        let response = handle_command(cmd, &state).await;

        // Send response
        let response_str = serde_json::to_string(&response)? + "\n";
        writer.write_all(response_str.as_bytes()).await?;
    }

    Ok(())
}

async fn handle_command(cmd: Command, state: &Arc<AppState>) -> CommandResponse {
    match cmd {
        Command::Start => {
            log::info!("Received START command");
            CommandResponse::success(serde_json::json!({"state": "running"}))
        }
        Command::Stop => {
            log::info!("Received STOP command");
            CommandResponse::success(serde_json::json!({"state": "stopped"}))
        }
        Command::Stats => {
            log::info!("Received STATS command");

            // Get system info
            let mut sys = state.system.write().await;
            sys.refresh_all();

            let uptime = state.start_time.elapsed().as_secs();
            let cpu_usage = sys.global_cpu_info().cpu_usage();
            let mem_total = sys.total_memory() / 1024 / 1024; // MB
            let mem_used = sys.used_memory() / 1024 / 1024; // MB
            let packets = *state.packets_sent.read().await;

            let stats = serde_json::json!({
                "uptime_seconds": uptime,
                "cpu_usage_percent": cpu_usage,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "packets_sent": packets,
            });

            CommandResponse::success(stats)
        }
        Command::GetConfig => {
            log::info!("Received GET_CONFIG command");
            CommandResponse::success(serde_json::json!({
                "udp_target": "127.0.0.1:9999",
                "tcp_bind": "0.0.0.0:8888",
                "ipc_socket": "/tmp/yolo_rust.sock"
            }))
        }
    }
}
