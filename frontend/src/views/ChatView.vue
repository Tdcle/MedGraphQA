<template>
  <a-layout class="layout">
    <a-layout-sider class="sider" :width="280">
      <div class="sidebar-shell">
        <div class="brand">MedGraphQA</div>

        <a-button type="primary" long :loading="creatingSession" :disabled="loading" @click="newConversation">新对话</a-button>

        <div class="session-list">
          <div
            v-for="item in sessions"
            :key="item.conversation_id"
            :class="['session-item', item.conversation_id === conversationId ? 'active' : '']"
            @click="openConversation(item.conversation_id)"
          >
            <div class="session-main">
              <span class="session-title">{{ item.title || '未命名会话' }}</span>
              <span class="session-time">{{ formatTime(item.updated_at) }}</span>
              <span v-if="item.awaiting_user_clarification" class="session-status">待补充</span>
            </div>
            <button
              class="session-delete"
              title="删除会话"
              @click.stop="deleteConversation(item)"
            >
              ×
            </button>
          </div>
          <div v-if="!sessions.length" class="empty-sessions">暂无会话</div>
        </div>

        <div class="sidebar-footer">
          <div class="footer-top">
            <div class="user-info">
              <img class="footer-avatar" :src="userAvatar" alt="用户头像" />
              <div class="user-copy">
                <div class="username">{{ user?.username || '未登录' }}</div>
                <div class="role">{{ user?.is_admin ? '管理员' : '普通用户' }}</div>
              </div>
            </div>
            <a-badge :count="pendingMemoryCount" :dot="pendingMemoryCount > 0">
              <a-button size="small" @click="openMemoryDrawer">记忆</a-button>
            </a-badge>
          </div>
          <a-button status="danger" long @click="logout">退出登录</a-button>
        </div>
      </div>
    </a-layout-sider>

    <a-layout-content class="content">
      <a-card class="chat-card" :bordered="false">
        <div class="messages">
          <div v-for="(item, idx) in messages" :key="idx" :class="['message', item.role]">
            <img class="message-avatar" :src="avatarForRole(item.role)" :alt="item.role === 'user' ? '用户头像' : '助手头像'" />
            <div class="message-body">
              <div class="message-name">{{ item.role === 'user' ? (user?.username || '用户') : 'MedGraphQA' }}</div>
              <div class="bubble">
                <template v-if="item.loading">
                  <div v-if="item.text" class="text">{{ item.text }}</div>
                  <div class="loading-line">
                    <a-spin :size="16" />
                    <span>{{ item.status || (item.text ? '正在生成回答' : '正在处理') }}</span>
                  </div>
                </template>
                <div v-else class="text">{{ item.text }}</div>
                <template v-if="showDebugMeta && item.meta">
                  <details class="debug-meta">
                    <summary>调试信息</summary>
                    <div class="meta">
                      <div><b>识别实体：</b>{{ item.meta.entities }}</div>
                      <div><b>识别意图：</b>{{ item.meta.intents }}</div>
                    </div>
                  </details>
                </template>
              </div>
            </div>
          </div>
        </div>

        <div class="input-wrap">
          <a-textarea
            v-model="query"
            class="chat-input"
            :disabled="loading"
            :auto-size="{ minRows: 2, maxRows: 4 }"
            placeholder="请输入医疗问题，按 Enter 发送"
            @keydown.enter.exact.prevent="sendMessage"
          />
        </div>
      </a-card>
    </a-layout-content>

    <a-drawer
      v-model:visible="memoryDrawerVisible"
      title="记忆管理"
      :width="420"
      unmount-on-close
      @open="loadMemories"
    >
      <div class="memory-section">
        <div class="memory-heading">待确认</div>
        <div v-if="pendingMemories.length" class="memory-list">
          <div v-for="item in pendingMemories" :key="item.id" class="memory-item">
            <a-textarea
              v-model="memoryDrafts[item.id]"
              :auto-size="{ minRows: 2, maxRows: 4 }"
            />
            <div class="memory-meta">{{ memoryTypeLabel(item.memory_type) }} · 置信度 {{ formatConfidence(item.confidence) }}</div>
            <a-space>
              <a-button type="primary" size="small" @click="activateMemory(item)">生效</a-button>
              <a-button size="small" @click="updateMemory(item)">修改</a-button>
              <a-button status="danger" size="small" @click="deleteMemory(item)">删除</a-button>
            </a-space>
          </div>
        </div>
        <a-empty v-else description="暂无待确认记忆" />
      </div>

      <a-divider />

      <div class="memory-section">
        <div class="memory-heading">已生效</div>
        <div v-if="activeMemories.length" class="memory-list">
          <div v-for="item in activeMemories" :key="item.id" class="memory-item active-memory">
            <a-textarea
              v-model="memoryDrafts[item.id]"
              :auto-size="{ minRows: 2, maxRows: 4 }"
            />
            <div class="memory-meta">{{ memoryTypeLabel(item.memory_type) }} · {{ item.source }}</div>
            <a-space>
              <a-button size="small" @click="updateMemory(item)">保存修改</a-button>
              <a-button status="danger" size="small" @click="deleteMemory(item)">删除</a-button>
            </a-space>
          </div>
        </div>
        <a-empty v-else description="暂无已生效记忆" />
      </div>
    </a-drawer>
  </a-layout>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { Message } from '@arco-design/web-vue'
import client from '../api/client'
import { clearAuth, getToken, getUser } from '../stores/auth'
import botAvatar from '../assets/bot.png'
import userAvatar from '../assets/user.png'

const router = useRouter()
const user = ref(getUser())
const query = ref('')
const loading = ref(false)
const creatingSession = ref(false)
const conversationId = ref(null)
const sessions = ref([])
const messages = ref([])
const memories = ref([])
const memoryDrafts = ref({})
const memoryDrawerVisible = ref(false)

const pendingMemories = computed(() => memories.value.filter((item) => item.status === 'pending'))
const activeMemories = computed(() => memories.value.filter((item) => item.status === 'active'))
const pendingMemoryCount = computed(() => pendingMemories.value.length)
const showDebugMeta = computed(() => Boolean(user.value?.is_admin))

const welcomeMessage = {
  role: 'assistant',
  text: '你好，我是医疗问答助手。请描述你的问题。',
}

function resetMessages() {
  messages.value = [welcomeMessage]
}

function avatarForRole(role) {
  return role === 'user' ? userAvatar : botAvatar
}

function formatTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString()
}

function formatEntities(entities) {
  if (!Array.isArray(entities) || entities.length === 0) return '-'
  return entities
    .map((item) => {
      const score = Number.isFinite(Number(item.score))
        ? Number(item.score).toFixed(2)
        : '-'
      return `${item.entity_type}:${item.canonical_name}（${item.mention}->${item.matched_alias}，${formatMatchMethod(item.match_method)}，匹配分${score}）`
    })
    .join('、')
}

function formatMatchMethod(method) {
  if (!method) return '-'
  const methodLabels = {
    postgres_exact: '别名精确匹配',
    elasticsearch: '全文检索',
    elasticsearch_vector: '向量召回',
    kg_inference: '图谱关联推断',
    disease_resolution: '疾病置信度推断',
  }
  if (!method.startsWith('rrf:')) return methodLabels[method] || method
  const parts = method
    .replace('rrf:', '')
    .split('+')
    .map((part) => methodLabels[part] || part)
  return `融合召回：${parts.join('+')}`
}

function metadataToMeta(metadata) {
  if (!metadata || !metadata.entities && !metadata.intents) return null
  return {
    intents: (metadata.intents || []).join('、') || '-',
    entities: formatEntities(metadata.entities),
  }
}

function memoryTypeLabel(type) {
  const labels = {
    allergy: '过敏史',
    pregnancy: '孕期',
    chronic_disease: '慢病史',
    medication: '用药',
    profile: '基础信息',
    preference: '偏好',
    recent_symptom: '近期症状',
  }
  return labels[type] || type
}

function formatConfidence(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : '-'
}

function syncMemoryDrafts() {
  const drafts = {}
  for (const item of memories.value) drafts[item.id] = item.text
  memoryDrafts.value = drafts
}

async function loadSessions() {
  const { data } = await client.get('/chat/sessions')
  sessions.value = data || []
}

async function loadMemories() {
  const { data } = await client.get('/memories')
  memories.value = data || []
  syncMemoryDrafts()
}

async function openMemoryDrawer() {
  memoryDrawerVisible.value = true
  try {
    await loadMemories()
  } catch (error) {
    Message.error(error?.response?.data?.detail || '加载记忆失败')
  }
}

async function activateMemory(item) {
  try {
    await updateMemory(item, false)
    await client.post(`/memories/${item.id}/activate`)
    Message.success('记忆已生效')
    await loadMemories()
  } catch (error) {
    Message.error(error?.response?.data?.detail || '操作失败')
  }
}

async function updateMemory(item, showSuccess = true) {
  const text = (memoryDrafts.value[item.id] || '').trim()
  if (!text) {
    Message.warning('记忆内容不能为空')
    return
  }
  await client.put(`/memories/${item.id}`, { text })
  if (showSuccess) Message.success('已保存')
  await loadMemories()
}

async function deleteMemory(item) {
  try {
    await client.delete(`/memories/${item.id}`)
    Message.success('已删除')
    await loadMemories()
  } catch (error) {
    Message.error(error?.response?.data?.detail || '删除失败')
  }
}

async function openConversation(id) {
  conversationId.value = id
  const { data } = await client.get(`/chat/sessions/${id}/messages`)
  messages.value = (data || []).map((item) => ({
    role: item.role,
    text: item.content,
    meta: item.role === 'assistant' ? metadataToMeta(item.metadata) : null,
  }))
  if (!messages.value.length) resetMessages()
}

async function deleteConversation(item) {
  try {
    await client.delete(`/chat/sessions/${item.conversation_id}`)
    if (item.conversation_id === conversationId.value) {
      conversationId.value = null
      query.value = ''
      resetMessages()
    }
    await loadSessions()
    Message.success('会话已删除')
  } catch (error) {
    Message.error(error?.response?.data?.detail || '删除会话失败')
    if (error?.response?.status === 401) await logout()
  }
}

async function newConversation() {
  if (creatingSession.value) return
  creatingSession.value = true
  try {
    const { data } = await client.post('/chat/sessions')
    conversationId.value = data.conversation_id
    query.value = ''
    resetMessages()
    await loadSessions()
  } catch (error) {
    Message.error(error?.response?.data?.detail || '创建会话失败')
    if (error?.response?.status === 401) await logout()
  } finally {
    creatingSession.value = false
  }
}

async function logout() {
  try {
    await client.post('/auth/logout')
  } catch {
    // Local cleanup is enough if the token is already invalid.
  }
  conversationId.value = null
  clearAuth()
  router.push('/login')
}

async function sendMessage() {
  const content = query.value.trim()
  if (!content) return
  messages.value.push({ role: 'user', text: content })
  const assistantMessage = {
    role: 'assistant',
    text: '',
    loading: true,
    status: '正在连接服务',
    meta: null,
  }
  messages.value.push(assistantMessage)
  query.value = ''
  loading.value = true
  try {
    const data = await sendMessageStream({
      query: content,
      conversation_id: conversationId.value,
    })
    conversationId.value = data.conversation_id || conversationId.value
    assistantMessage.loading = false
    assistantMessage.status = ''
    assistantMessage.text = data.answer
    assistantMessage.meta = {
      intents: (data.intents || []).join('、') || '-',
      entities: formatEntities(data.entities),
    }
    await loadSessions()
    await loadMemories()
  } catch (error) {
    const idx = messages.value.indexOf(assistantMessage)
    if (idx >= 0) messages.value.splice(idx, 1)
    const detail = error?.response?.data?.detail || error?.message || '请求失败'
    Message.error(detail)
    if (error?.response?.status === 401) {
      await logout()
    }
  } finally {
    loading.value = false
  }
}

function apiUrl(path) {
  const base = import.meta.env.VITE_API_BASE || '/api'
  return `${base.replace(/\/$/, '')}${path}`
}

async function sendMessageStream(payload) {
  const token = getToken()
  const response = await fetch(apiUrl('/chat/ask/stream'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(payload),
  })
  if (response.status === 401) {
    clearAuth()
    throw { response: { status: 401, data: { detail: '登录状态已失效' } } }
  }
  if (!response.ok || !response.body) {
    throw new Error('请求失败')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalData = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const part of parts) {
      const event = parseSseEvent(part)
      if (!event) continue
      if (event.event === 'status') {
        const last = messages.value[messages.value.length - 1]
        if (last?.loading) last.status = event.data.message
      } else if (event.event === 'token') {
        const last = messages.value[messages.value.length - 1]
        if (last?.loading) {
          last.text += event.data.text || ''
          last.status = '正在生成回答'
        }
      } else if (event.event === 'final') {
        finalData = event.data
      } else if (event.event === 'error') {
        throw new Error(event.data.message || '请求失败')
      }
    }
  }
  if (!finalData) throw new Error('响应为空')
  return finalData
}

function parseSseEvent(raw) {
  const lines = raw.split('\n')
  let event = 'message'
  let data = ''
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) data += line.slice(5).trim()
  }
  if (!data) return null
  return { event, data: JSON.parse(data) }
}

onMounted(async () => {
  resetMessages()
  try {
    await loadSessions()
    await loadMemories()
  } catch (error) {
    if (error?.response?.status === 401) await logout()
  }
})
</script>

<style scoped>
.layout {
  height: 100vh;
}

.sider {
  height: 100vh;
  background: #ffffff;
  border-right: 1px solid #e5e6eb;
}

.sider :deep(.arco-layout-sider-children) {
  height: 100%;
  overflow: hidden;
}

.sidebar-shell {
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-height: 0;
  padding: 18px 14px;
}

.brand {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 14px;
}

.session-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  margin-top: 14px;
  padding-right: 2px;
}

.session-item {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 28px;
  align-items: center;
  gap: 6px;
  text-align: left;
  border: 0;
  border-radius: 8px;
  padding: 9px 10px;
  margin-bottom: 6px;
  color: #1d2129;
  background: transparent;
  cursor: pointer;
}

.session-main {
  display: grid;
  gap: 3px;
  min-width: 0;
}

.session-item:hover,
.session-item.active {
  background: #f2f3f5;
}

.session-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
}

.session-delete {
  width: 24px;
  height: 24px;
  border: 0;
  border-radius: 8px;
  color: #86909c;
  background: transparent;
  cursor: pointer;
  font-size: 18px;
  line-height: 22px;
  opacity: 0;
}

.session-item:hover .session-delete {
  opacity: 1;
}

.session-delete:hover {
  color: #cb2634;
  background: #ffece8;
}

.session-time,
.session-status,
.empty-sessions,
.role {
  color: #86909c;
  font-size: 12px;
}

.session-status {
  color: #d25f00;
}

.sidebar-footer {
  margin-top: auto;
  border-top: 1px solid #e5e6eb;
  padding-top: 14px;
  display: grid;
  gap: 10px;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 0;
}

.user-copy {
  display: grid;
  gap: 3px;
  min-width: 0;
}

.footer-avatar {
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  border-radius: 8px;
  object-fit: cover;
  background: #f2f3f5;
  border: 1px solid #e5e6eb;
}

.username {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.footer-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.content {
  padding: 16px;
}

.chat-card {
  height: calc(100vh - 32px);
  border-radius: 8px;
  display: flex;
}

.chat-card :deep(.arco-card-body) {
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: 12px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 4px;
}

.message {
  display: flex;
  margin-bottom: 10px;
  gap: 9px;
  align-items: flex-start;
}

.message.user {
  justify-content: flex-start;
  flex-direction: row-reverse;
}

.message.assistant {
  justify-content: flex-start;
}

.message-avatar {
  width: 34px;
  height: 34px;
  flex: 0 0 auto;
  border-radius: 8px;
  object-fit: cover;
  background: #ffffff;
  border: 1px solid #e5e6eb;
}

.message-body {
  display: grid;
  gap: 4px;
  max-width: min(860px, 82%);
  min-width: 0;
}

.message.user .message-body {
  justify-items: end;
}

.message-name {
  color: #86909c;
  font-size: 12px;
  line-height: 18px;
}

.bubble {
  border-radius: 8px;
  padding: 10px 12px;
  background: #f2f3f5;
  white-space: pre-wrap;
  word-break: break-word;
}

.message.user .bubble {
  background: #e8f3ff;
}

.meta {
  font-size: 12px;
  color: #4e5969;
}

.debug-meta {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #e5e6eb;
  white-space: normal;
}

.debug-meta summary {
  width: fit-content;
  cursor: pointer;
  color: #4e5969;
  font-size: 12px;
  line-height: 18px;
  user-select: none;
}

.debug-meta .meta {
  margin-top: 6px;
  display: grid;
  gap: 4px;
  white-space: pre-wrap;
  word-break: break-word;
}

.loading-line {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #4e5969;
}

.input-wrap {
  display: flex;
  justify-content: center;
  width: 100%;
  padding: 4px 0 2px;
}

.input-wrap :deep(.arco-textarea-wrapper) {
  width: min(920px, 100%);
  border: 1px solid #1d2129 !important;
  border-radius: 16px !important;
  background: #ffffff !important;
  box-shadow: 0 10px 28px rgba(29, 33, 41, 0.16) !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

.input-wrap :deep(.arco-textarea-wrapper:hover) {
  border-color: #1d2129 !important;
  background: #ffffff !important;
  box-shadow: 0 12px 32px rgba(29, 33, 41, 0.18) !important;
}

.input-wrap :deep(.arco-textarea-wrapper:focus-within),
.input-wrap :deep(.arco-textarea-wrapper.arco-textarea-focus) {
  border-color: #1d2129 !important;
  background: #ffffff !important;
  box-shadow: 0 14px 36px rgba(29, 33, 41, 0.22) !important;
}

.input-wrap :deep(.arco-textarea) {
  padding: 13px 16px !important;
  line-height: 1.6;
  font-size: 14px;
  color: #1d2129;
  background: transparent !important;
}

.input-wrap :deep(.arco-textarea::placeholder) {
  color: #a9b2bf;
}

.memory-section {
  display: grid;
  gap: 10px;
}

.memory-heading {
  font-weight: 600;
}

.memory-list {
  display: grid;
  gap: 10px;
}

.memory-item {
  display: grid;
  gap: 8px;
  border: 1px solid #e5e6eb;
  border-radius: 8px;
  padding: 10px;
  background: #ffffff;
}

.active-memory {
  background: #f7f8fa;
}

.memory-meta {
  color: #86909c;
  font-size: 12px;
}
</style>
