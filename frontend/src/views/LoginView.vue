<template>
  <div class="login-page">
    <el-card class="login-card">
      <div class="login-header">
        <div class="login-logo">🎓</div>
        <h2>EduAgent</h2>
        <p>AI 教学辅助系统</p>
      </div>
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
        @submit.prevent="handleLogin"
      >
        <el-form-item label="用户名" prop="username">
          <el-input
            v-model="form.username"
            placeholder="请输入用户名"
            size="large"
            :prefix-icon="User"
          />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            size="large"
            :prefix-icon="Lock"
            show-password
            @keyup.enter="handleLogin"
          />
        </el-form-item>
        <el-button
          type="primary"
          size="large"
          :loading="loading"
          style="width: 100%; margin-top: 8px"
          @click="handleLogin"
        >
          登录
        </el-button>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { authApi } from '@/api/auth'
import { useAuthStore } from '@/stores/auth'
import type { AxiosError } from 'axios'

const router = useRouter()
const auth = useAuthStore()

const formRef = ref<FormInstance>()
const loading = ref(false)
const form = reactive({ username: '', password: '' })

const rules: FormRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

async function handleLogin() {
  await formRef.value?.validate()
  loading.value = true
  try {
    const { data } = await authApi.login({
      username: form.username,
      password: form.password,
    })
    auth.login(data.access_token, {
      userId: data.user_id,
      role: data.role as 'student' | 'teacher' | 'admin',
      tenantId: '',
      username: form.username,
    })
    router.push('/dashboard')
  } catch (err) {
    // 401 是后端明确返回的（用户名/密码不对、账号禁用），拦截器不会弹消息，这里显示；
    // 其他状态码拦截器已弹过 detail；无 response = 网络错误/跨域拦截，单独提示
    const e = err as AxiosError<{ detail?: string }>
    if (e.response?.status === 401) {
      ElMessage.error(e.response.data?.detail ?? '用户名或密码错误')
    } else if (!e.response) {
      ElMessage.error('无法连接服务器，请确认后端已启动（http://localhost:8006）')
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.login-card {
  width: 400px;
  border-radius: 12px;
}
.login-header {
  text-align: center;
  margin-bottom: 24px;
}
.login-logo {
  font-size: 48px;
  margin-bottom: 8px;
}
.login-header h2 {
  margin: 0 0 4px;
  font-size: 24px;
  color: #1a1a1a;
}
.login-header p {
  margin: 0;
  color: #8c8c8c;
  font-size: 14px;
}
</style>
