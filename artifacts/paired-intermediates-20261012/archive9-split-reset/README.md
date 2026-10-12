# Archive9 split: AMD graphics timeout

The frozen Mac-input/AMD-output reference run used ordinary archive9 `8cae56b0748a5c66add985122d8bbf72b7492e2e0828f713f1b0d7425a59a4e5` and test-delivery `23bf318378e5e115fc642b41ebf92d0d3eec72b4b31c2414b1f8331facab6dcc`. Expectations were committed in500bcc64 before execution. Both residents opened and their actual resolved sessions were observed through the installed factory. The19-token and21-token requests completed their two frozen steps each.

The184-token request failed before its first output. Its row preserves `mapAsync: Buffer was destroyed before mapping was resolved` during pool destruction. The subsequent pool snapshot and settlement checks also failed, masking that error in the controller's final exception. The full failed receipt is preserved, including both errors and incomplete settlement. No full55 split result or successful cleanup is claimed.

The bounded kernel journal identifies a graphics ring timeout at2026-10-12 17:07:48 EDT for our owned Chrome GPU process798293, followed by a successful ring reset and `device wedged, but recovered through reset`. This is observed hardware/driver recovery, not proof of a particular faulty shader or out-of-memory condition. The read-only process record binds that GPU process to our Chrome798245 and controller798132. The isolated resident controller67359 exited1; owned remote Chrome controller798132 was then explicitly stopped. Public contributor processes were not restarted or modified; a hardware reset can affect other GPU clients.

Both unsplit archive9 provider runs passed independently before this split. Next diagnosis must preserve the primary device-loss event and actual last submission/dispatch geometry, then replay only the same184-token case under a coordinated window. No arithmetic, reference, budget or input changes are authorized by this failed receipt.

Component: `doppler.runtime-source.inference.pipelines.text` evidence and Reploid verification fixture. Intent: preserved. This receipt does not establish signed opening, network acquisition or application release acceptance.
