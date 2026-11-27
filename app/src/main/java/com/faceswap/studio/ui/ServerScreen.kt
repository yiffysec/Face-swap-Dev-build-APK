package com.faceswap.studio.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.Dns
import androidx.compose.material.icons.filled.Error
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.faceswap.studio.data.ProcessingMode
import com.faceswap.studio.engine.ServerStatus
import com.faceswap.studio.ui.theme.Good
import com.faceswap.studio.ui.theme.Warn

/** Settings → Processing sub-page. */
@Composable
fun ServerSettings(vm: AppViewModel) {
    SectionTitle("Where processing runs")
    Panel {
        ServerModePicker(vm)
    }
    val s by vm.settings.collectAsState()
    if (s.processingMode == ProcessingMode.REMOTE) {
        SectionTitle("Server")
        Panel { ServerConnectionForm(vm) }
        Note("The setup script prints the address, port and key to enter here once it finishes on the server. " +
            "InSwapper isn't offered in this mode – HyperSwap (the default) and everything else work exactly as on-device.",
            Modifier.padding(top = 10.dp))
    }
}

/** Shown in place of the Models screen when remote mode is on and nothing is connected yet (first-run gate). */
@Composable
fun ServerGateScreen(vm: AppViewModel) {
    Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp)) {
        ScreenHeader("Connect to your server", "FaceSwap Studio needs a running inference server in Remote mode")
        Panel {
            Note("Run server/setup_server.sh on your machine, then enter the address it prints below. " +
                "Switch back to on-device processing in Settings → Processing if you'd rather not run a server.")
            Spacer(Modifier.height(10.dp))
            ServerConnectionForm(vm)
        }
    }
}

@Composable
private fun ServerModePicker(vm: AppViewModel) {
    val s by vm.settings.collectAsState()
    Segmented(ProcessingMode.entries.map { it to it.label }, s.processingMode) { mode ->
        vm.updateSettings { it.copy(processingMode = mode) }
    }
    Note(s.processingMode.hint)
}

@Composable
private fun ServerConnectionForm(vm: AppViewModel) {
    val s by vm.settings.collectAsState()
    val status by vm.serverStatus.collectAsState()
    fun up(t: (com.faceswap.studio.data.AppSettings) -> com.faceswap.studio.data.AppSettings) = vm.updateSettings(t)

    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = s.serverHost, onValueChange = { up { st -> st.copy(serverHost = it) } },
            label = { Text("Host or IP") }, singleLine = true, modifier = Modifier.weight(2f),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
        )
        OutlinedTextField(
            value = if (s.serverPort == 0) "" else s.serverPort.toString(),
            onValueChange = { v -> up { st -> st.copy(serverPort = v.filter(Char::isDigit).take(5).toIntOrNull() ?: 0) } },
            label = { Text("Port") }, singleLine = true, modifier = Modifier.width(96.dp),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        )
    }
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
        Switch(checked = s.serverUseTls, onCheckedChange = { up { st -> st.copy(serverUseTls = it) } })
        Spacer(Modifier.width(8.dp))
        Column { Text("Use TLS (https)", style = MaterialTheme.typography.bodyMedium); Note("Turn off only for a plain-HTTP test server on a trusted local network.") }
    }
    OutlinedTextField(
        value = s.serverApiKey, onValueChange = { up { st -> st.copy(serverApiKey = it) } },
        label = { Text("API key (optional)") }, singleLine = true, modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
    )

    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 12.dp)) {
        Button(onClick = vm::connectToServer, enabled = status !is ServerStatus.Connecting, modifier = Modifier.weight(1f).height(48.dp)) {
            if (status is ServerStatus.Connecting) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp) else Text("Connect")
        }
        if (status is ServerStatus.Connected) OutlinedButton(onClick = vm::disconnectServer, modifier = Modifier.height(48.dp)) { Text("Disconnect") }
    }
    Spacer(Modifier.height(10.dp))
    ServerStatusRow(status)
}

@Composable
private fun ServerStatusRow(status: ServerStatus) {
    when (status) {
        is ServerStatus.Disconnected -> Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.CloudOff, null, Modifier.height(18.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.width(6.dp)); Text("Not connected", style = MaterialTheme.typography.bodyMedium)
        }
        is ServerStatus.Connecting -> Text("Connecting…", style = MaterialTheme.typography.bodyMedium)
        is ServerStatus.Connected -> Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.CheckCircle, null, Modifier.height(18.dp), tint = Good)
                Spacer(Modifier.width(6.dp))
                Text("${status.info.name} ${status.info.version}", style = MaterialTheme.typography.bodyMedium)
            }
            Note("${status.info.device} · ${status.info.models.size} models available", Modifier.padding(start = 24.dp))
        }
        is ServerStatus.Failed -> Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.Error, null, Modifier.height(18.dp), tint = Warn)
            Spacer(Modifier.width(6.dp)); Text(status.message, style = MaterialTheme.typography.bodyMedium, color = Warn)
        }
    }
}

/** Small icon used by the Settings hub's "Processing" nav row. */
@Composable
fun ProcessingHubIcon() = Icon(Icons.Default.Dns, null)
