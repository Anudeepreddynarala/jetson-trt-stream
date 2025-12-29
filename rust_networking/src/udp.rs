use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use anyhow::Result;
use crate::protocol::DetectionMessage;

pub struct UdpStreamer {
    socket: UdpSocket,
    target_addr: String,
    packets_sent: u64,
}

impl UdpStreamer {
    pub async fn new(bind_addr: &str, target_addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind(bind_addr).await?;
        log::info!("UDP streamer bound to {}", bind_addr);
        log::info!("UDP target: {}", target_addr);

        Ok(Self {
            socket,
            target_addr: target_addr.to_string(),
            packets_sent: 0,
        })
    }

    pub async fn send_detection(&mut self, msg: &DetectionMessage) -> Result<()> {
        let json = serde_json::to_string(msg)?;

        // Send UDP packet
        self.socket.send_to(json.as_bytes(), &self.target_addr).await?;
        self.packets_sent += 1;

        if self.packets_sent % 100 == 0 {
            log::debug!("UDP: Sent {} packets", self.packets_sent);
        }

        Ok(())
    }

    pub fn get_packets_sent(&self) -> u64 {
        self.packets_sent
    }
}

pub async fn run_udp_streamer(
    mut rx: mpsc::Receiver<DetectionMessage>,
    bind_addr: String,
    target_addr: String,
) -> Result<()> {
    let mut streamer = UdpStreamer::new(&bind_addr, &target_addr).await?;

    log::info!("UDP streamer running");

    while let Some(msg) = rx.recv().await {
        if let Err(e) = streamer.send_detection(&msg).await {
            log::error!("Failed to send UDP: {}", e);
        }
    }

    log::info!("UDP streamer shutdown (sent {} packets)", streamer.get_packets_sent());
    Ok(())
}
