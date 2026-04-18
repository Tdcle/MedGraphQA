<template>
  <div class="page">
    <a-card class="panel" :bordered="false">
      <h2 class="title">MedGraphQA</h2>
      <a-tabs v-model:active-key="activeTab" type="rounded">
        <a-tab-pane key="login" title="登录">
          <a-form :model="loginForm" layout="vertical" @submit.prevent>
            <a-form-item field="username" label="用户名">
              <a-input v-model="loginForm.username" placeholder="请输入用户名" @press-enter="onLogin" />
            </a-form-item>
            <a-form-item field="password" label="密码">
              <a-input-password v-model="loginForm.password" placeholder="请输入密码" @press-enter="onLogin" />
            </a-form-item>
            <a-button type="primary" long :loading="loading" @click="onLogin">登录</a-button>
          </a-form>
        </a-tab-pane>
        <a-tab-pane key="register" title="注册">
          <a-form :model="registerForm" layout="vertical" @submit.prevent>
            <a-form-item field="username" label="用户名">
              <a-input v-model="registerForm.username" placeholder="3-32 位用户名" @press-enter="onRegister" />
            </a-form-item>
            <a-form-item field="password" label="密码">
              <a-input-password v-model="registerForm.password" placeholder="至少 6 位密码" @press-enter="onRegister" />
            </a-form-item>
            <a-button type="primary" long :loading="loading" @click="onRegister">注册</a-button>
          </a-form>
        </a-tab-pane>
      </a-tabs>
    </a-card>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { Message } from '@arco-design/web-vue'
import { useRouter } from 'vue-router'
import client from '../api/client'
import { saveAuth } from '../stores/auth'

const router = useRouter()
const activeTab = ref('login')
const loading = ref(false)
const loginForm = reactive({ username: '', password: '' })
const registerForm = reactive({ username: '', password: '' })

function validateLogin() {
  if (!loginForm.username.trim()) return '请输入用户名'
  if (!loginForm.password) return '请输入密码'
  return ''
}

function validateRegister() {
  const username = registerForm.username.trim()
  if (username.length < 3 || username.length > 32) return '用户名长度需要在 3-32 位之间'
  if (registerForm.password.length < 6) return '密码至少需要 6 位'
  if (registerForm.password.length > 128) return '密码不能超过 128 位'
  return ''
}

function errorDetail(error, fallback) {
  const detail = error?.response?.data?.detail
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg).join('；') || fallback
  }
  return detail || fallback
}

async function onLogin() {
  const validationError = validateLogin()
  if (validationError) {
    Message.error(validationError)
    return
  }
  loading.value = true
  try {
    const { data } = await client.post('/auth/login', {
      username: loginForm.username.trim(),
      password: loginForm.password,
    })
    saveAuth(data.access_token, { username: data.username, is_admin: data.is_admin })
    Message.success('登录成功')
    router.push('/chat')
  } catch (error) {
    Message.error(errorDetail(error, '登录失败'))
  } finally {
    loading.value = false
  }
}

async function onRegister() {
  const validationError = validateRegister()
  if (validationError) {
    Message.error(validationError)
    return
  }
  loading.value = true
  try {
    await client.post('/auth/register', {
      username: registerForm.username.trim(),
      password: registerForm.password,
    })
    Message.success('注册成功，请登录')
    activeTab.value = 'login'
    loginForm.username = registerForm.username.trim()
    loginForm.password = ''
  } catch (error) {
    Message.error(errorDetail(error, '注册失败'))
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 16px;
}

.panel {
  width: 100%;
  max-width: 420px;
  border-radius: 8px;
}

.title {
  margin: 0 0 20px;
  text-align: center;
}
</style>
