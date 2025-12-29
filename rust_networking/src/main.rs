mod protocol;
mod udp;
mod tcp;

use tokio::sync::mpsc;
use tokio::net::UnixListener;
use tokio::io::{AsyncBufReadExt, BufReader};
use anyhow::Result;
use std::sync::Arc;

use protocol::DetectionMessage;
use tcp::AppState;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Starting YOLO-Rust-TRT networking layer");

    // Create shared state
    let state = Arc::new(AppState::new());

    // Channel for Python → UDP forwarding
    let (tx, rx) = mpsc::channel(1000);

    // Spawn UDP streamer
    let udp_handle = tokio::spawn(async move {
        udp::run_udp_streamer(
            rx,
            "0.0.0.0:0".to_string(),
            "127.0.0.1:9999".to_string(),
        )
        .await
    });

    // Spawn TCP command server
    let tcp_state = state.clone();
    let tcp_handle = tokio::spawn(async move {
        match tcp::TcpCommandServer::new("0.0.0.0:8888", tcp_state).await {
            Ok(server) => {
                if let Err(e) = server.run().await {
                    log::error!("TCP server error: {}", e);
                }
            }
            Err(e) => {
                log::error!("Failed to start TCP server: {}", e);
            }
        }
    });

    // Unix socket listener for Python IPC
    let socket_path = "/tmp/yolo_rust.sock";

    // Remove old socket file if exists
    let _ = std::fs::remove_file(socket_path);

    let listener = UnixListener::bind(socket_path)?;
    log::info!("Unix socket listening at {}", socket_path);

    // Accept Python connection
    let ipc_handle = tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    log::info!("Python client connected");
                    let tx = tx.clone();
                    let state = state.clone();

                    tokio::spawn(async move {
                        let mut reader = BufReader::new(stream);
                        let mut line = String::new();
                        let mut frame_count = 0u64;

                        loop {
                            line.clear();
                            match reader.read_line(&mut line).await {
                                Ok(0) => {
                                    log::info!("Python client disconnected (frames: {})", frame_count);
                                    break;
                                }
                                Ok(_) => {
                                    // Parse detection message
                                    match serde_json::from_str::<DetectionMessage>(&line) {
                                        Ok(msg) => {
                                            frame_count += 1;

                                            // Update packets sent counter
                                            {
                                                let mut packets = state.packets_sent.write().await;
                                                *packets += 1;
                                            }

                                            // Forward to UDP
                                            if let Err(e) = tx.send(msg).await {
                                                log::error!("Failed to forward message: {}", e);
                                                break;
                                            }

                                            if frame_count % 100 == 0 {
                                                log::debug!("Processed {} frames", frame_count);
                                            }
                                        }
                                        Err(e) => {
                                            log::warn!("Invalid JSON from Python: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("Read error: {}", e);
                                    break;
                                }
                            }
                        }
                    });
                }
                Err(e) => {
                    log::error!("Accept error: {}", e);
                }
            }
        }
    });

    // Wait for tasks (they run forever until interrupted)
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            log::info!("Received shutdown signal");
        }
        result = udp_handle => {
            log::error!("UDP task exited: {:?}", result);
        }
        result = tcp_handle => {
            log::error!("TCP task exited: {:?}", result);
        }
        result = ipc_handle => {
            log::error!("IPC task exited: {:?}", result);
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(socket_path);
    log::info!("Shutdown complete");

    Ok(())
}
