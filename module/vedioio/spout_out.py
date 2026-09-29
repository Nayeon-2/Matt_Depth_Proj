from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np


class SpoutOutputSender:
    """Send the latest BGR or BGRA frame to Spout from an owned OpenGL thread.

    BGR frames are sent with alpha=255. BGRA frames keep their alpha channel
    (straight alpha), so receivers can use it as an opacity mask.
    """

    dl = False
    use_sleep = False

    def __init__(self, name: str, width: int, height: int, *, invert: bool = False):
        self.name = str(name).strip() or "RVM_OUTPUT"
        self.width = int(width)
        self.height = int(height)
        self.invert = bool(invert)
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._alive = False
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self.frames_sent = 0
        self.frames_dropped = 0
        self.transport_reported = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self._alive = True
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"SpoutOutput[{self.name}]",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self.stop()
            raise RuntimeError("Spout sender initialization timed out")
        if self._error is not None:
            raise RuntimeError(f"Spout sender initialization failed: {self._error}") from self._error
        return self

    def submit(self, frame_bgr: np.ndarray):
        if not self._alive or self._error is not None:
            return
        if frame_bgr is None or frame_bgr.shape[:2] != (self.height, self.width):
            return
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] not in (3, 4):
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            self.frames_dropped += 1

    def stats(self) -> dict[str, int]:
        return {
            "spout_sent": int(self.frames_sent),
            "spout_dropped": int(self.frames_dropped),
        }

    def stop(self):
        self._alive = False
        try:
            if self._queue.full():
                self._queue.get_nowait()
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run(self):
        glfw = None
        sender = None
        texture = None
        window = None
        try:
            import glfw as _glfw
            from OpenGL import GL
            from SpoutGL import SpoutSender

            glfw = _glfw
            if not glfw.init():
                raise RuntimeError("glfw.init() failed")
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            window = glfw.create_window(
                self.width, self.height, "RVM Spout Sender", None, None
            )
            if not window:
                raise RuntimeError("hidden OpenGL window creation failed")
            glfw.make_context_current(window)
            glfw.swap_interval(0)

            texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                self.width,
                self.height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                None,
            )

            sender = SpoutSender()
            sender.setSenderName(self.name)
            self._ready.set()

            while self._alive or not self._queue.empty():
                try:
                    frame = self._queue.get(timeout=0.25)
                except queue.Empty:
                    glfw.poll_events()
                    continue
                if frame is None:
                    break

                conversion = (
                    cv2.COLOR_BGRA2RGBA if frame.shape[2] == 4 else cv2.COLOR_BGR2RGBA
                )
                rgba = cv2.cvtColor(np.ascontiguousarray(frame), conversion)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glTexSubImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    self.width,
                    self.height,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    rgba,
                )
                GL.glFlush()
                ok = sender.sendTexture(
                    int(texture),
                    int(GL.GL_TEXTURE_2D),
                    self.width,
                    self.height,
                    self.invert,
                    0,
                )
                if ok:
                    self.frames_sent += 1
                    if not self.transport_reported:
                        print(
                            "[INFO] Spout transport: "
                            f"gldx_ready={sender.isGLDXReady()} "
                            f"gldx={sender.getGLDX()} cpu={sender.getCPU()} "
                            f"cpu_share={sender.getCPUshare()}",
                            flush=True,
                        )
                        self.transport_reported = True
                glfw.poll_events()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            print(f"[WARN] Spout sender stopped: {exc}", flush=True)
        finally:
            try:
                if sender is not None:
                    sender.releaseSender()
            except Exception:
                pass
            try:
                if texture is not None:
                    from OpenGL import GL

                    GL.glDeleteTextures([texture])
            except Exception:
                pass
            try:
                if glfw is not None and window is not None:
                    glfw.destroy_window(window)
                if glfw is not None:
                    glfw.terminate()
            except Exception:
                pass
            self._alive = False
